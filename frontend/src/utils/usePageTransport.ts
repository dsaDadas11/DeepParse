import { useMount } from 'ahooks'
import { useState } from 'react'

type PageTransportKey<T> = symbol & { readonly __type?: T }

const tempMap = new Map<symbol, unknown>()

/**
 * 用于页面间数据传输
 * 需要注意的是，仅在组件初始化时有效
 */
export function usePageTransport<T>(key: PageTransportKey<T>) {
  const [data, setData] = useState<T | undefined>(() => tempMap.get(key) as T | undefined)

  useMount(() => {
    const tempData = tempMap.get(key) as T | undefined
    setData(tempData)
    tempMap.delete(key)
  })

  return {
    data,
    setData,
  }
}

export function setPageTransport<T>(key: PageTransportKey<T>, data: T) {
  tempMap.set(key, data)
}
