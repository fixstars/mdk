document.addEventListener("DOMContentLoaded", function async() {
    requestIdleCallback(() => {
        document.querySelector(".fill-container-button").addEventListener("click", () => {
            const element = document.querySelector(".container");
            const computedStyle = window.getComputedStyle(element);
            computedStyle.maxWidth == '1320px' ? element.style.maxWidth = '100%' : element.style.maxWidth = '1320px'
        });
    })
});
