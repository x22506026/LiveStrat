const SIDEBAR_STORAGE_KEY = 'livestrat_sidebar_collapsed';

function applySidebarState(collapsed) {
    document.body.classList.toggle('sidebar-collapsed', collapsed);

    const sidebarToggle = document.getElementById('sidebar-toggle');
    const topbarSidebarToggle = document.getElementById('topbar-sidebar-toggle');

    if (sidebarToggle) {
        sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
        sidebarToggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    }

    if (topbarSidebarToggle) {
        topbarSidebarToggle.setAttribute('aria-expanded', String(!collapsed));
        topbarSidebarToggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    }
}

function readSidebarState() {
    try {
        return window.localStorage.getItem(SIDEBAR_STORAGE_KEY) === 'true';
    } catch (error) {
        return false;
    }
}

function writeSidebarState(collapsed) {
    try {
        window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(collapsed));
    } catch (error) {
        return;
    }
}

function toggleSidebarState() {
    const nextCollapsed = !document.body.classList.contains('sidebar-collapsed');
    applySidebarState(nextCollapsed);
    writeSidebarState(nextCollapsed);
}

applySidebarState(readSidebarState());

const sidebarToggle = document.getElementById('sidebar-toggle');
if (sidebarToggle) {
    sidebarToggle.addEventListener('click', toggleSidebarState);
}

const topbarSidebarToggle = document.getElementById('topbar-sidebar-toggle');
if (topbarSidebarToggle) {
    topbarSidebarToggle.addEventListener('click', toggleSidebarState);
}

const signoutModal = document.getElementById('signout-modal');
const signoutTrigger = document.getElementById('sidebar-signout-trigger');
const signoutCancel = document.getElementById('signout-cancel');

if (signoutModal) {
    signoutModal.hidden = true;
}

function setSignoutModalOpen(open) {
    if (!signoutModal) {
        return;
    }
    signoutModal.hidden = !open;
    document.body.classList.toggle('modal-open', open);
}

if (signoutTrigger) {
    signoutTrigger.addEventListener('click', () => {
        setSignoutModalOpen(true);
    });
}

if (signoutCancel) {
    signoutCancel.addEventListener('click', () => {
        setSignoutModalOpen(false);
    });
}

if (signoutModal) {
    signoutModal.addEventListener('click', (event) => {
        if (event.target === signoutModal) {
            setSignoutModalOpen(false);
        }
    });
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && signoutModal && !signoutModal.hidden) {
        setSignoutModalOpen(false);
    }
});
