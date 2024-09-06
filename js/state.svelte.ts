export class BorderBoxSize {
  borderBoxSize: ResizeObserverSize[] = $state([]);

  width: number = $derived(
    this.borderBoxSize.length ? this.borderBoxSize[0].inlineSize : 0,
  );

  height: number = $derived(
    this.borderBoxSize.length ? this.borderBoxSize[0].blockSize : 0,
  );

  constructor(borderBoxSize: ResizeObserverSize[]) {
    this.borderBoxSize = borderBoxSize;
  }
}

function getRootDiv() {
  let value: HTMLDivElement | null = $state(null);

  return {
    get value() {
      return value;
    },
    set value(v: HTMLDivElement | null) {
      value = v;
    },
  };
}

export let widgetDimensions = new BorderBoxSize([]);
export let rootDiv = getRootDiv();
