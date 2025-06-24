// Polyfills for NEAR wallet selector and other crypto libraries
import { Buffer } from 'buffer';

// Make Buffer available globally
if (typeof globalThis !== 'undefined') {
  globalThis.Buffer = Buffer;
} else if (typeof window !== 'undefined') {
  (window as any).Buffer = Buffer;
} else {
  // @ts-ignore
  global.Buffer = Buffer;
}

// Export for explicit imports
export { Buffer };
