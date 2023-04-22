import { initUIComponents } from "./components/components";
import log from "./log/logger";

export const init = async () => {
  log("😃 Initializing HWEmoji ")
  await initUIComponents();
  log("💪 HWEmoji initialized")
}

