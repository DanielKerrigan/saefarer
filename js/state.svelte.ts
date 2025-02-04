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

export let rootDiv = getRootDiv();
