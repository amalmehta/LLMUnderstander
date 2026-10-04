import 'preact'
declare module 'preact' {
  namespace JSX {
    // Allow CSS custom properties (e.g. style={{ '--sec': 'var(--sec-agents)' }}).
    interface CSSProperties { [key: `--${string}`]: string | number | undefined }
  }
}
