class RootElement {
  value: HTMLElement;

  constructor(value: HTMLElement) {
    this.value = $state(value);
  }
}

export let root: RootElement;

export function setupState(element: HTMLElement) {
  root = new RootElement(element);
}
