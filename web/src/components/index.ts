import { Dropdown } from 'materialize-css';

export const initUIComponents = async (): Promise<void> => {
    await initMaterializeCssComponents()
}

const initMaterializeCssComponents = (): Promise<void> => {
    return new Promise(resolve => {
        document.addEventListener('DOMContentLoaded', function() {
            var elems = document.querySelectorAll('.dropdown-trigger');
            Dropdown.init(elems);
            initEmojiSelector();
            resolve();
        });
    });
}

const initEmojiSelector = () => {

}