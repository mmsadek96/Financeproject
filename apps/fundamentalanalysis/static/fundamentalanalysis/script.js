document.addEventListener("DOMContentLoaded", function () {
    const collapsibleButton = document.querySelector('.collapsible');
    const collapsibleContent = document.querySelector('.collapsible-content');

    if (collapsibleButton && collapsibleContent) {
        collapsibleButton.addEventListener('click', () => {
            const isVisible = collapsibleContent.style.display === 'block';
            collapsibleContent.style.display = isVisible ? 'none' : 'block';
            collapsibleButton.textContent = isVisible ? 'View Full Analysis' : 'Hide Analysis';
        });
    }

    // If you need to run any other JS logic (e.g. dynamic chart updates), add it here.
});
