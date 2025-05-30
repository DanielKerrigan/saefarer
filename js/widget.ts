import "./style.css";
import Widget from "./components/Widget.svelte";
import type { DataModel } from "./types";
import type { Render } from "@anywidget/types";
import { mount, unmount } from "svelte";
import { setupSyncedState } from "./synced-state.svelte";
import { setupState } from "./state.svelte";

const render: Render<DataModel> = ({ model, el }) => {
  setupSyncedState(model);
  setupState(el);

  let widget = mount(Widget, { target: el });
  return () => unmount(widget);
};

export default { render };
