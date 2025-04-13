export function clamp<T>(x: T, floor: T, ceil: T): T {
  if (x < floor) {
    return floor;
  } else if (x > ceil) {
    return ceil;
  } else {
    return x;
  }
}
