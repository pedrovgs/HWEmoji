import { Dropdown } from "materialize-css";
import { gemoji, Gemoji } from "gemoji";
import log from "../log/logger";
import { WorkMode, ComponentsListeners } from "../domain/model";

export const initUIComponents = async (listeners: ComponentsListeners): Promise<void> => {
  await initMaterializeCssComponents(listeners);
};

const initMaterializeCssComponents = (listeners: ComponentsListeners): Promise<void> => {
  return new Promise((resolve) => {
    document.addEventListener("DOMContentLoaded", function () {
      var elems = document.querySelectorAll(".dropdown-trigger");
      Dropdown.init(elems);
      initEmojiSelector(listeners);
      initModeSelector(listeners);
      resolve();
    });
  });
};

const initEmojiSelector = (listeners: ComponentsListeners) => {
  const selector = document.getElementById("emoji-selector");
  const selectorTitle = document.getElementById("emoji-selector-title");
  if (selector != null && selectorTitle != null) {
    gemoji.forEach((emojiInfo) => {
      const emojiOption = document.createElement("li");
      const emojiName = emojiInfo.description.charAt(0).toUpperCase() + emojiInfo.description.substring(1);
      emojiOption.innerHTML = `<a>${emojiInfo.emoji}  ${emojiName}</a>`;
      selector.appendChild(emojiOption);
      emojiOption.onclick = () => {
        log(`Emoji selected ${emojiInfo.emoji} - ${emojiName}`);
        selectorTitle.textContent = `Emoji to draw: ${emojiInfo.emoji}`;
        listeners.onEmojiSelected(emojiInfo);
      };
    });
  }
  log(`✅ Emoji selector configured`);
};

const initModeSelector = (listeners: ComponentsListeners) => {
  const modes = [
    { mode: WorkMode.collectData, description: "💪 Collect Data" },
    { mode: WorkMode.test, description: "✅ Test model" },
  ];
  const selector = document.getElementById("mode-selector");
  const selectorTitle = document.getElementById("mode-selector-title");
  if (selector != null && selectorTitle != null) {
    modes.forEach((workMode) => {
      const modeOption = document.createElement("li");
      modeOption.innerHTML = `<a>${workMode.description}</a>`;
      selector.appendChild(modeOption);
      modeOption.onclick = () => {
        log(`Mode selected: ${workMode.description}`);
        selectorTitle.textContent = `Mode option: ${workMode.description}`;
        listeners.onModeSelected(workMode.mode);
      };
    });
  }
  log(`✅ Mode selector configured`);
};
