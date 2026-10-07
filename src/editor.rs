//! The [`Editor`] trait implementation for Vizia editors.

use crossbeam::atomic::AtomicCell;
use nice_plug_core::context::gui::GuiContext;
use nice_plug_core::debug::*;
use nice_plug_core::editor::dpi::{LogicalSize, NativeSize, Size};
use nice_plug_core::editor::{
    Editor, EditorHandle, HostMethods, Modifiers, ParentWindowHandle, ResizeHint, SpawnedEditor,
    VirtualKeyCode,
};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use vizia::prelude::*;
use vizia::views::TextEvent;

use crate::widgets::RawParamEvent;
use crate::widgets::param_registry::ParamRegistry;
use crate::{ViziaState, ViziaTheming, widgets};

/// Bridges nice-plug's `ParentWindowHandle` (raw-window-handle 0.6) to the
/// raw-window-handle 0.5 `HasRawWindowHandle` that vizia's forked baseview
/// expects, by extracting the platform handle per variant.
struct RwhParent(ParentWindowHandle);

unsafe impl raw_window_handle::HasRawWindowHandle for RwhParent {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        use raw_window_handle::{
            AppKitWindowHandle, RawWindowHandle, Win32WindowHandle, XcbWindowHandle,
            XlibWindowHandle,
        };
        match self.0 {
            ParentWindowHandle::AppKitNsView(ns_view) => {
                let mut h = AppKitWindowHandle::empty();
                h.ns_view = ns_view.as_ptr();
                RawWindowHandle::AppKit(h)
            }
            ParentWindowHandle::Win32Hwnd(hwnd) => {
                let mut h = Win32WindowHandle::empty();
                h.hwnd = hwnd.get() as *mut std::ffi::c_void;
                RawWindowHandle::Win32(h)
            }
            ParentWindowHandle::XlibWindow(window) => {
                let mut h = XlibWindowHandle::empty();
                h.window = window;
                RawWindowHandle::Xlib(h)
            }
            ParentWindowHandle::XcbWindow(window) => {
                let mut h = XcbWindowHandle::empty();
                h.window = window.get();
                RawWindowHandle::Xcb(h)
            }
        }
    }
}

/// A key-down event queued by the host-thread
/// `Editor::on_virtual_key_from_host` callback, waiting for the next
/// `on_idle` tick to dispatch on the GUI thread.
///
/// Split between character input (goes through `TextEvent::InsertText`)
/// and non-printable control keys (goes through `WindowEvent::KeyDown`)
/// so the GUI-thread drain stays trivially correct without having to
/// re-guess a key's semantics: the classification is made on the host
/// thread from the host's virtual key code.
pub(crate) enum KeyInject {
    /// A printable character (derived from a virtual key that maps 1:1
    /// to a printable character: Space, Numpad0..Numpad9, the numpad
    /// operator keys, Equals). Dispatched as `TextEvent::InsertText`.
    Char(char),
    /// A non-printable key (Backspace, Enter, Tab, Escape, arrows,
    /// Home/End, Delete, F-keys, etc.). Dispatched as
    /// `WindowEvent::KeyDown(code, Some(key))` so the target view's
    /// own key handling (e.g. textbox's `WindowEvent::KeyDown`
    /// match arm) runs.
    ControlKey(Code, Key),
}

/// State shared between [`ViziaEditor`] (invoked from the host UI
/// thread) and the vizia `on_idle` callback (invoked on the GUI thread).
///
/// The `Editor::on_virtual_key_from_host` callback runs on the host
/// thread and cannot reach into the live vizia `Context`. Instead, it
/// consults `text_focused` (kept in sync from `on_idle`) to decide
/// whether to claim the key, and pushes a [`KeyInject`] into `pending`.
/// The next `on_idle` tick drains the queue and dispatches each entry
/// to the currently focused entity.
pub(crate) struct KeyInjectState {
    /// `true` while the vizia focused view reports element name `"textbox"`.
    /// Updated on every `on_idle` tick.
    text_focused: AtomicBool,
    /// Keys the host delivered via the virtual-key hook that still need
    /// to be dispatched into the focused view on the next idle tick.
    pending: Mutex<VecDeque<KeyInject>>,
}

impl KeyInjectState {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            text_focused: AtomicBool::new(false),
            pending: Mutex::new(VecDeque::new()),
        })
    }
}

/// An [`Editor`] implementation that calls a vizia draw loop.
pub struct ViziaEditor {
    pub(crate) vizia_state: Arc<ViziaState>,
    /// The user's app function.
    pub(crate) app: Arc<dyn Fn(&mut Context, GuiContext) + 'static + Send + Sync>,
    /// What level of theming to apply. See [`ViziaEditorTheming`].
    pub(crate) theming: ViziaTheming,

    /// The scaling factor reported by the host, if any. On macOS this will never be set and we
    /// should use the system scaling factor instead.
    pub(crate) scaling_factor: AtomicCell<Option<f32>>,

    /// Whether to emit a parameters changed event during the next idle callback. This is set in the
    /// `parameter_values_changed()` implementation and it can be used by widgets to explicitly
    /// check for new parameter values. This is useful when the parameter value is (indirectly) used
    /// to compute a property in an event handler. Like when positioning an element based on the
    /// display value's width.
    pub(crate) emit_parameters_changed_event: Arc<AtomicBool>,

    /// Shared registry of `SyncSignal<f32>`s tracking each parameter's live value. Widgets
    /// subscribe via `cx.data::<ParamRegistry>()`; the editor calls `flush_all()` from the
    /// `parameter_value_changed` / `parameter_values_changed` hooks so the reactive graph picks
    /// up value changes that nice-plug reports.
    pub(crate) param_registry: ParamRegistry,

    /// Shared state bridging the host-thread
    /// `on_virtual_key_from_host` callback with the GUI-thread
    /// `on_idle` callback. See [`KeyInjectState`].
    pub(crate) key_inject: Arc<KeyInjectState>,
}

impl Editor for ViziaEditor {
    type Handle = ViziaEditorHandle;

    fn spawn(
        &self,
        parent: Option<ParentWindowHandle>,
        _wait_for_parent: bool,
        fallback_scale_factor: Option<f64>,
        gui_context: GuiContext,
        _host: Option<HostMethods>,
    ) -> Result<SpawnedEditor<Self::Handle>, Box<dyn Error>> {
        let app = self.app.clone();
        let vizia_state = self.vizia_state.clone();
        let theming = self.theming;
        let param_registry = self.param_registry.clone();
        let param_registry_for_build = param_registry.clone();

        // Prevent stale runtime-bound signals from previous editor instances from being reused.
        param_registry.clear_signals();

        let (unscaled_width, unscaled_height) = vizia_state.inner_logical_size();
        if let Some(sf) = fallback_scale_factor {
            self.scaling_factor.store(Some(sf as f32));
        }
        let system_scaling_factor = self.scaling_factor.load();
        let user_scale_factor = vizia_state.user_scale_factor();

        // Resizing is zoom-only (aspect-locked user scale): vizia_baseview can
        // rescale its window but not resize it freely. Requests from the host
        // (`set_size`) or a dragged standalone window land here and are
        // applied on the UI thread in `on_idle`.
        let pending_scale = Arc::new(AtomicCell::new(None::<f64>));
        let standalone = parent.is_none();
        // Native sizes are logical on macOS, physical elsewhere.
        let native_scale = if cfg!(target_os = "macos") {
            1.0
        } else {
            system_scaling_factor.unwrap_or(1.0) as f64
        };

        // `GuiContext` is a cheap (Arc-backed) struct in 0.4.2; clone per consumer.
        let context = gui_context;
        let mut application = Application::new(Box::new(move |cx: &mut Context| {
            // Set some default styles to match the iced integration
            //if theming >= ViziaTheming::Custom {
            // NOTE: `Context::set_default_font` was removed upstream as a deprecated API
            // (vizia commit ff943a0b, "Context: remove deprecated APIs and clarify docs").
            // The default font is now controlled through stylesheets — `theme.css` below
            // can set `* { font-family: ...; }` if a specific font is required.
            if let Err(err) = cx.add_stylesheet(include_style!("src/assets/theme.css")) {
                nice_error!("Failed to load stylesheet: {err:?}");
                panic!();
            }

            // There doesn't seem to be any way to bundle styles with a widget, so we'll always
            // include the style sheet for our custom widgets at context creation
            widgets::register_theme(cx);
            //}

            // Install the parameter signal registry so widgets can find it via
            // `cx.data::<ParamRegistry>()`. `ParamRegistry` is a cheap handle (Arc internally),
            // so the editor keeps a clone for flushing on parameter changes.
            param_registry_for_build.clone().build(cx);

            // Any widget can change the parameters by emitting `ParamEvent` events. This model will
            // handle them automatically.
            widgets::ParamModel {
                context: context.clone(),
            }
            .build(cx);

            // And we'll link `WindowEvent::ResizeWindow` and `WindowEvent::SetScale` events to our
            // `ViziaState`. We'll notify the host when any of these change.
            let current_inner_window_size = EventContext::new(cx).cache.get_bounds(Entity::root());
            widgets::WindowModel {
                context: context.clone(),
                vizia_state: vizia_state.clone(),
                last_inner_window_size: AtomicCell::new((
                    current_inner_window_size.width() as u32,
                    current_inner_window_size.height() as u32,
                )),
            }
            .build(cx);

            app(cx, context.clone())
        }) as AppBuilder)
        .with_scale_policy(
            system_scaling_factor
                .map(|factor| WindowScalePolicy::ScaleFactor(factor as f64))
                .unwrap_or(WindowScalePolicy::SystemScaleFactor),
        )
        .inner_size((unscaled_width, unscaled_height))
        .user_scale_factor(user_scale_factor)
        .on_idle({
            let emit_parameters_changed_event = self.emit_parameters_changed_event.clone();
            let key_inject = self.key_inject.clone();
            let pending_scale = pending_scale.clone();
            let vizia_state = self.vizia_state.clone();
            move |cx| {
                #[cfg(target_os = "macos")]
                if standalone {
                    let base = vizia_state.inner_logical_size();
                    if let Some(w) = standalone_content_width(base)
                        && (w - vizia_state.scaled_logical_size().0 as f64).abs() > 1.0
                    {
                        pending_scale.store(Some(w / base.0 as f64));
                    }
                }
                #[cfg(not(target_os = "macos"))]
                let _ = standalone;
                if let Some(scale) = pending_scale.take() {
                    vizia_state.set_user_scale_factor(scale);
                    cx.emit_custom(Event::new(WindowEvent::SetUserScale(scale)));
                }

                if emit_parameters_changed_event
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    cx.emit_custom(
                        Event::new(RawParamEvent::ParametersChanged)
                            .propagate(Propagation::Subtree),
                    );
                }

                // Keep `text_focused` in sync so the host-thread
                // `on_virtual_key_from_host` callback can decide
                // synchronously whether to claim a key. The element
                // name `"textbox"` is set by
                // `vizia::views::Textbox::element()`.
                key_inject
                    .text_focused
                    .store(cx.focused_element() == Some("textbox"), Ordering::Release);

                // Drain any keys queued by `on_virtual_key_from_host`
                // and dispatch them to the focused view. Buffered
                // inside a short-lived lock to keep the critical
                // section bounded.
                //
                // `on_virtual_key_from_host` has already classified
                // each item on the host thread using the VST3
                // `key_code`, so the drain just mechanically
                // dispatches each variant: chars via
                // `TextEvent::InsertText`, control keys via
                // `WindowEvent::KeyDown` so the focused view's own
                // key handler (e.g. textbox's `KeyDown` match arm)
                // runs.
                let drained: Vec<KeyInject> = {
                    let mut q = key_inject.pending.lock().unwrap_or_else(|e| e.into_inner());
                    q.drain(..).collect()
                };
                if !drained.is_empty() {
                    let mut ec = EventContext::new(cx);
                    let target = ec.focused();
                    for entry in drained {
                        match entry {
                            KeyInject::Char(c) => {
                                ec.emit_to(target, TextEvent::InsertText(c.to_string()));
                            }
                            KeyInject::ControlKey(code, key) => {
                                ec.emit_to(target, WindowEvent::KeyDown(code, Some(key)));
                            }
                        }
                    }
                }
            }
        });

        // This way the plugin can decide to use none of the built in theming
        if theming == ViziaTheming::None {
            application = application.ignore_default_theme();
        }

        // Hosts give a parent; the standalone target doesn't, so hand the
        // unopened app back for `run_until_closed` to run blocking.
        let (window, app) = match parent {
            Some(p) => (Some(application.open_parented(&RwhParent(p))), None),
            None => (None, Some(application)),
        };

        self.vizia_state.open.store(true, Ordering::Release);
        Ok(SpawnedEditor {
            handle: ViziaEditorHandle {
                vizia_state: self.vizia_state.clone(),
                window,
                pending_scale,
                native_scale,
                param_registry,
                emit_parameters_changed_event: self.emit_parameters_changed_event.clone(),
                key_inject: self.key_inject.clone(),
            },
            window: app,
        })
    }

    fn size(&self) -> NativeSize<u32> {
        let (w, h) = self.vizia_state.scaled_logical_size();
        let scale = self.scaling_factor.load().unwrap_or(1.0) as f64;
        NativeSize::from_size(
            nice_plug_core::editor::dpi::Size::Logical(
                nice_plug_core::editor::dpi::LogicalSize::new(w as f64, h as f64),
            ),
            scale,
        )
    }

    fn resize_hint(&self) -> ResizeHint {
        zoom_hint(&self.vizia_state)
    }
}

/// Aspect-locked resizing down to half the base size (see `pending_scale`).
fn zoom_hint(state: &ViziaState) -> ResizeHint {
    let (w, h) = state.inner_logical_size();
    ResizeHint {
        // `with_aspect_ratio` sets the ratio but not this flag.
        preserve_aspect_ratio: true,
        ..ResizeHint::resizable()
            .with_aspect_ratio(w, h)
            .with_min_logical_size(LogicalSize::new(w as f32 / 2.0, h as f32 / 2.0))
    }
}

/// Standalone only: baseview creates a fixed-size `NSWindow`, so make it
/// resizable (aspect-locked) and return its current content width.
#[cfg(target_os = "macos")]
fn standalone_content_width((w, h): (u32, u32)) -> Option<f64> {
    use objc2_app_kit::{NSApp, NSWindowStyleMask};
    use objc2_foundation::{MainThreadMarker, NSSize};
    let win = NSApp(MainThreadMarker::new()?).windows().firstObject()?;
    let mask = win.styleMask();
    if !mask.contains(NSWindowStyleMask::Resizable) {
        win.setStyleMask(mask | NSWindowStyleMask::Resizable);
        win.setContentAspectRatio(NSSize::new(w as f64, h as f64));
        win.setContentMinSize(NSSize::new(w as f64 / 2.0, h as f64 / 2.0));
    }
    Some(win.contentView()?.frame().size.width)
}

/// Boxed so the unopened standalone `Application` has a nameable type.
type AppBuilder = Box<dyn Fn(&mut Context) + Send>;

impl EditorHandle for ViziaEditorHandle {
    type Window = Option<Application<AppBuilder>>;
    type Error = Infallible;

    fn run_until_closed(window: Self::Window) -> Result<(), Self::Error> {
        // `ApplicationError` is uninhabited.
        window.map_or(Ok(()), |app| app.run().map_err(|e| match e {}))
    }

    fn set_parent(
        &self,
        _parent: ParentWindowHandle,
        _window: &Self::Window,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn show(&self, _window: &Self::Window) -> Result<(), Self::Error> {
        Ok(())
    }

    fn hide(&self, _window: &Self::Window) -> Result<(), Self::Error> {
        Ok(())
    }

    fn host_main_thread_callback(&self, _window: &Self::Window) {}

    fn set_size(
        &self,
        new_size: NativeSize<u32>,
        _window: &Self::Window,
    ) -> Result<(), Self::Error> {
        let base_width = self.vizia_state.inner_logical_size().0 as f64;
        self.pending_scale
            .store(Some(new_size.width as f64 / self.native_scale / base_width));
        Ok(())
    }

    fn set_fallback_scale_factor(
        &self,
        _scale_factor: f64,
        _window: &Self::Window,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn adjust_size(
        &self,
        new_size: NativeSize<u32>,
        _window: &Self::Window,
    ) -> Option<NativeSize<u32>> {
        let (w, h) = self.vizia_state.scaled_logical_size();
        let current = NativeSize::from_size(
            Size::Logical(LogicalSize::new(w as f64, h as f64)),
            self.native_scale,
        );
        Some(zoom_hint(&self.vizia_state).adjust_size(new_size, current, self.native_scale))
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        // Push the new value into the registry's signals — observers bound via `Binding::new`
        // wake up and rebuild. Also flag a `ParametersChanged` idle event for any widgets that
        // still rely on the older (pre-signal) notification path.
        self.param_registry.flush_all();
        self.emit_parameters_changed_event
            .store(true, Ordering::Relaxed);
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {
        self.param_registry.flush_all();
        self.emit_parameters_changed_event
            .store(true, Ordering::Relaxed);
    }

    fn on_virtual_key_from_host(
        &self,
        key_code: VirtualKeyCode,
        is_down: bool,
        modifiers: Modifiers,
    ) -> bool {
        // Called from the host's UI thread (e.g. REAPER dispatching
        // `IPlugView::onKeyDown` / `onKeyUp`). Claim the key only when
        // a textbox is currently focused; otherwise the host's
        // accelerator (e.g. space -> transport) should run normally.
        if !self.key_inject.text_focused.load(Ordering::Acquire) {
            return false;
        }

        // Modifier-held combinations (Cmd+A, Cmd+Left, Shift+Arrow,
        // Option+Backspace, etc.) are claimed by the host or handled by
        // AppKit's `keyDown:` + `doCommandBySelector:` path where
        // vizia's textbox reads modifier state for line/word movement.
        // Dispatching through our injection queue here would double-fire
        // and lose modifier context. Return `false` so the host keeps
        // the key and AppKit's normal path runs.
        if !modifiers.is_empty() {
            return false;
        }

        // Classify the virtual key. The host hands us virtual keys that
        // split into two groups vizia consumes differently:
        //
        // - Keys that represent a printable character (Space, numpad
        //   digits/operators, `=`) go in as `TextEvent::InsertText`.
        // - Named control keys (Backspace, Enter, arrows, F-keys,
        //   etc.) go in as `WindowEvent::KeyDown(code, Some(key))` so
        //   the focused view's own key handler (textbox's `KeyDown`
        //   match arm for Backspace / Enter / arrows) runs.
        //
        // Keys we don't enumerate here (media / volume keys, Select,
        // Print, modifier-only presses, Super) fall through to
        // `return false` so the host's own binding runs.
        let inject = match key_code {
            VirtualKeyCode::Space => Some(KeyInject::Char(' ')),
            VirtualKeyCode::Numpad0 => Some(KeyInject::Char('0')),
            VirtualKeyCode::Numpad1 => Some(KeyInject::Char('1')),
            VirtualKeyCode::Numpad2 => Some(KeyInject::Char('2')),
            VirtualKeyCode::Numpad3 => Some(KeyInject::Char('3')),
            VirtualKeyCode::Numpad4 => Some(KeyInject::Char('4')),
            VirtualKeyCode::Numpad5 => Some(KeyInject::Char('5')),
            VirtualKeyCode::Numpad6 => Some(KeyInject::Char('6')),
            VirtualKeyCode::Numpad7 => Some(KeyInject::Char('7')),
            VirtualKeyCode::Numpad8 => Some(KeyInject::Char('8')),
            VirtualKeyCode::Numpad9 => Some(KeyInject::Char('9')),
            VirtualKeyCode::NumpadMultiply => Some(KeyInject::Char('*')),
            VirtualKeyCode::NumpadAdd => Some(KeyInject::Char('+')),
            VirtualKeyCode::NumpadSeparator => Some(KeyInject::Char(',')),
            VirtualKeyCode::NumpadSubtract => Some(KeyInject::Char('-')),
            VirtualKeyCode::NumpadDecimal => Some(KeyInject::Char('.')),
            VirtualKeyCode::NumpadDivide => Some(KeyInject::Char('/')),
            VirtualKeyCode::Equals => Some(KeyInject::Char('=')),

            VirtualKeyCode::Backspace => {
                Some(KeyInject::ControlKey(Code::Backspace, Key::Backspace))
            }
            VirtualKeyCode::Tab => Some(KeyInject::ControlKey(Code::Tab, Key::Tab)),
            VirtualKeyCode::Return => Some(KeyInject::ControlKey(Code::Enter, Key::Enter)),
            VirtualKeyCode::NumpadEnter => {
                Some(KeyInject::ControlKey(Code::NumpadEnter, Key::Enter))
            }
            VirtualKeyCode::Pause => Some(KeyInject::ControlKey(Code::Pause, Key::Pause)),
            VirtualKeyCode::Escape => Some(KeyInject::ControlKey(Code::Escape, Key::Escape)),
            VirtualKeyCode::End => Some(KeyInject::ControlKey(Code::End, Key::End)),
            VirtualKeyCode::Home => Some(KeyInject::ControlKey(Code::Home, Key::Home)),
            VirtualKeyCode::ArrowLeft => {
                Some(KeyInject::ControlKey(Code::ArrowLeft, Key::ArrowLeft))
            }
            VirtualKeyCode::ArrowUp => Some(KeyInject::ControlKey(Code::ArrowUp, Key::ArrowUp)),
            VirtualKeyCode::ArrowRight => {
                Some(KeyInject::ControlKey(Code::ArrowRight, Key::ArrowRight))
            }
            VirtualKeyCode::ArrowDown => {
                Some(KeyInject::ControlKey(Code::ArrowDown, Key::ArrowDown))
            }
            VirtualKeyCode::PageUp => Some(KeyInject::ControlKey(Code::PageUp, Key::PageUp)),
            VirtualKeyCode::PageDown => Some(KeyInject::ControlKey(Code::PageDown, Key::PageDown)),
            VirtualKeyCode::Insert => Some(KeyInject::ControlKey(Code::Insert, Key::Insert)),
            VirtualKeyCode::Delete => Some(KeyInject::ControlKey(Code::Delete, Key::Delete)),
            VirtualKeyCode::F1 => Some(KeyInject::ControlKey(Code::F1, Key::F1)),
            VirtualKeyCode::F2 => Some(KeyInject::ControlKey(Code::F2, Key::F2)),
            VirtualKeyCode::F3 => Some(KeyInject::ControlKey(Code::F3, Key::F3)),
            VirtualKeyCode::F4 => Some(KeyInject::ControlKey(Code::F4, Key::F4)),
            VirtualKeyCode::F5 => Some(KeyInject::ControlKey(Code::F5, Key::F5)),
            VirtualKeyCode::F6 => Some(KeyInject::ControlKey(Code::F6, Key::F6)),
            VirtualKeyCode::F7 => Some(KeyInject::ControlKey(Code::F7, Key::F7)),
            VirtualKeyCode::F8 => Some(KeyInject::ControlKey(Code::F8, Key::F8)),
            VirtualKeyCode::F9 => Some(KeyInject::ControlKey(Code::F9, Key::F9)),
            VirtualKeyCode::F10 => Some(KeyInject::ControlKey(Code::F10, Key::F10)),
            VirtualKeyCode::F11 => Some(KeyInject::ControlKey(Code::F11, Key::F11)),
            VirtualKeyCode::F12 => Some(KeyInject::ControlKey(Code::F12, Key::F12)),

            _ => None,
        };

        let Some(entry) = inject else {
            return false;
        };

        // Vizia's text-input model is press-driven: TextEvent::InsertText
        // and the textbox's KeyDown handlers run on the press only. Push
        // the queued event on key-down; on key-up, just claim the event
        // so the host doesn't pick the release up as a separate
        // accelerator (BillyDM's reasoning on nice-plug#9).
        if is_down {
            self.key_inject
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push_back(entry);
        }
        true
    }
}

/// The window handle used for [`ViziaEditor`].
pub struct ViziaEditorHandle {
    vizia_state: Arc<ViziaState>,
    window: Option<WindowHandle>,
    pending_scale: Arc<AtomicCell<Option<f64>>>,
    native_scale: f64,
    param_registry: ParamRegistry,
    emit_parameters_changed_event: Arc<AtomicBool>,
    key_inject: Arc<KeyInjectState>,
}

/// The window handle enum stored within 'WindowHandle' contains raw pointers. Is there a way around
/// having this requirement?
unsafe impl Send for ViziaEditorHandle {}

impl Drop for ViziaEditorHandle {
    fn drop(&mut self) {
        self.vizia_state.open.store(false, Ordering::Release);
        // XXX: This should automatically happen when the handle gets dropped, but apparently not
        if let Some(w) = self.window.as_mut() {
            w.close();
        }
        self.param_registry.clear_signals();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_is_zoom_resizable() {
        let editor = crate::create_vizia_editor(
            ViziaState::new(|| (900, 500)),
            ViziaTheming::None,
            |_, _| {},
        )
        .unwrap();
        let hint = editor.resize_hint();
        assert!(hint.can_resize && hint.preserve_aspect_ratio);
        assert_eq!(
            (hint.aspect_ratio_width, hint.aspect_ratio_height),
            (900, 500)
        );
    }

    #[test]
    fn adjust_size_keeps_aspect_and_minimum() {
        let hint = zoom_hint(&ViziaState::new(|| (900, 500)));
        let cur = NativeSize::new(900, 500);
        assert_eq!(
            hint.adjust_size(NativeSize::new(1800, 700), cur, 1.0),
            NativeSize::new(1260, 700)
        );
        assert_eq!(
            hint.adjust_size(NativeSize::new(100, 100), cur, 1.0),
            NativeSize::new(450, 250)
        );
    }
}
