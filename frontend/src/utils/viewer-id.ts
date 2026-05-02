const VIEWER_ID_KEY = 'deepparse.viewer_id'
const VIEWER_ID_PATTERN = /^u_[a-f0-9]{32}$/
const FORCED_VIEWER_ID = import.meta.env.VITE_FORCE_VIEWER_ID

function generateHex(length: number) {
  const alphabet = '0123456789abcdef'
  let output = ''
  for (let index = 0; index < length; index += 1) {
    output += alphabet[Math.floor(Math.random() * alphabet.length)]
  }
  return output
}

function createViewerId() {
  const cryptoObject = globalThis.crypto
  if (cryptoObject?.randomUUID) {
    return `u_${cryptoObject.randomUUID().replace(/-/g, '')}`
  }

  return `u_${generateHex(32)}`
}

export function isViewerId(value: unknown): value is string {
  return typeof value === 'string' && VIEWER_ID_PATTERN.test(value)
}

export function getViewerId() {
  if (isViewerId(FORCED_VIEWER_ID)) {
    window.localStorage.setItem(VIEWER_ID_KEY, FORCED_VIEWER_ID)
    return FORCED_VIEWER_ID
  }

  const current = window.localStorage.getItem(VIEWER_ID_KEY)
  if (isViewerId(current)) {
    return current
  }

  const next = createViewerId()
  window.localStorage.setItem(VIEWER_ID_KEY, next)
  return next
}
