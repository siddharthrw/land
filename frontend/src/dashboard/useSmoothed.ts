import { useEffect, useRef, useState } from "react";

/**
 * Exponential smoothing for a rapidly-updating numeric readout. The backend
 * streams state at 50Hz, so a raw value like throttle visibly flickers
 * between nearby numbers every frame — this damps that into something a
 * human can actually read without lagging behind real changes.
 */
export function useSmoothed(value: number, alpha = 0.18): number {
  const ref = useRef(value);
  const [smoothed, setSmoothed] = useState(value);

  useEffect(() => {
    ref.current += (value - ref.current) * alpha;
    setSmoothed(ref.current);
  }, [value, alpha]);

  return smoothed;
}
