import "./style.css";
import Widget from "./components/Widget.svelte";
import { setupSyncedState } from "./synced-state.svelte";
import type { Model } from "./types";
import type { Render } from "@anywidget/types";
import { mount, unmount } from "svelte";

const render: Render<Model> = ({ model, el }) => {
  setupSyncedState(model);
  let widget = mount(Widget, { target: el });
  return () => unmount(widget);
};

export default { render };
