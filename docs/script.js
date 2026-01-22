function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    event.target.classList.add('active');
    // Trigger Plotly resize
    window.dispatchEvent(new Event('resize'));
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    sidebar.classList.toggle('active');
    overlay.classList.toggle('active');
}


function enterDashboard() {
    document.getElementById('landing-page').classList.add('hidden');
    document.getElementById('dashboard-view').style.display = 'block';

    // Trigger resize for Plotly charts as they are now visible
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 100);
}

function showIntro() {
    document.getElementById('landing-page').classList.remove('hidden');
}

