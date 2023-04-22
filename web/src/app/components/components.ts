import { Dropdown } from 'materialize-css';
import {gemoji} from "gemoji";
import log from '../log/logger';

export const initUIComponents = async (): Promise<void> => {
    await initMaterializeCssComponents()
}

const initMaterializeCssComponents = (): Promise<void> => {
    return new Promise(resolve => {
        document.addEventListener('DOMContentLoaded', function() {
            var elems = document.querySelectorAll('.dropdown-trigger');
            Dropdown.init(elems);
            initEmojiSelector();
            initModeSelector();
            resolve();
        });
    });
}

const initEmojiSelector = () => {
    const selector = document.getElementById("emoji-selector");
    const selectorTitle = document.getElementById("emoji-selector-title");
    if (selector != null && selectorTitle != null) {
        gemoji.forEach(emojiInfo => {
            const emojiOption = document.createElement("li");
            const emojiName = emojiInfo.description.charAt(0).toUpperCase() + emojiInfo.description.substring(1);
            emojiOption.innerHTML = `<a>${emojiInfo.emoji}  ${emojiName}</a>`
            selector.appendChild(emojiOption)    
            emojiOption.onclick = () => {
                log(`Emoji selected ${emojiInfo.emoji} - ${emojiName}`);
                selectorTitle.textContent = `Emoji to draw: ${emojiInfo.emoji}`;
            }
        });
    }
    log(`✅ Emoji selector configured`);
}

const initModeSelector = () => {
    const modes = ["💪 Generate Data", "✅ Test model"]
    const selector = document.getElementById("mode-selector");
    const selectorTitle = document.getElementById("mode-selector-title");
    if (selector != null && selectorTitle != null) {
        modes.forEach(mode => {
            const modeOption = document.createElement("li");
            modeOption.innerHTML = `<a>${mode}</a>`
            selector.appendChild(modeOption)    
            modeOption.onclick = () => {
                log(`Mode selected: ${mode}`);
                selectorTitle.textContent = `Mode option: ${mode}`;
            }
        });
    }
    log(`✅ Mode selector configured`);
}