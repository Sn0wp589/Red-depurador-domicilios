document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const platformTexts = [
        document.getElementById('current-platform-text'),
        document.getElementById('current-platform-process'),
        document.getElementById('btn-platform')
    ];
    let currentPlatform = 'DIDI';

    // File Drag & Drop
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.querySelector('.upload-content');
    const fileDetails = document.getElementById('file-details');
    const filenameDisplay = document.getElementById('filename');
    const removeBtn = document.getElementById('remove-file');
    const processBtn = document.getElementById('process-btn');

    let currentFile = null;

    // Tabs logic
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const platform = tab.dataset.tab.toUpperCase();
            currentPlatform = platform;
            
            platformTexts.forEach(el => {
                if(el) el.textContent = platform;
            });
            
            if(currentFile) removeFile();
        });
    });

    // Mode cards logic (Fusionar archivos)
    const modeCards = document.querySelectorAll('.mode-card');
    modeCards.forEach(card => {
        card.addEventListener('click', () => {
            if (card.querySelector('h3').textContent.includes('Fusionar')) {
                alert('La opción "Fusionar varios archivos" estará disponible en una próxima actualización. Por ahora, procesa tus archivos individualmente.');
            }
        });
    });

    // Dropzone logic
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--kfc-red)';
        dropZone.style.backgroundColor = '#fff0f0';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.backgroundColor = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.backgroundColor = '';
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeFile();
    });

    function handleFile(file) {
        currentFile = file;
        uploadContent.classList.add('hidden');
        fileDetails.classList.remove('hidden');
        filenameDisplay.textContent = file.name;
        
        processBtn.classList.remove('disabled');
        processBtn.classList.add('active');
    }

    function removeFile() {
        currentFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        fileDetails.classList.add('hidden');
        
        processBtn.classList.add('disabled');
        processBtn.classList.remove('active');
    }

    // Process File Logic (Fetch API to Python backend)
    processBtn.addEventListener('click', async () => {
        if (!currentFile || processBtn.classList.contains('disabled')) return;

        const loader = document.getElementById('loader');
        processBtn.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('platform', currentPlatform.toLowerCase());

        try {
            const response = await fetch('/api/index', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();

                // Extraer estadísticas
                document.getElementById('stat-filas').textContent = data.stats.total_filas || '0';
                document.getElementById('stat-datos').textContent = data.stats.datos_procesados || '0';
                document.getElementById('stat-sin-asignar').textContent = data.stats.sin_asignar || '0';
                document.getElementById('stat-tiendas').textContent = data.stats.tiendas_unicas || '0';
                
                const now = new Date();
                document.getElementById('update-time').textContent = `Hoy, ${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}`;

                // Download file from base64
                const byteCharacters = atob(data.file_base64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = data.filename || `${currentPlatform}_Procesado.xlsx`;
                document.body.appendChild(a);
                a.click();
                a.remove();

                // Render preview table
                if (data.preview_data) {
                    renderPreviewTable(data.preview_data);
                }

                // Save unassigned stores in global variable for modal
                window.currentUnassignedStores = data.unassigned_stores || [];
                
            } else {
                const errorData = await response.json();
                alert(`Error: ${errorData.detail || 'Ocurrió un error en el servidor.'}`);
            }
        } catch (error) {
            console.error(error);
            alert('Error de conexión. Asegúrate de que el backend en Python esté corriendo.');
        } finally {
            loader.classList.add('hidden');
            processBtn.classList.remove('hidden');
        }
    });

    function renderPreviewTable(dataArray) {
        const section = document.getElementById('preview-section');
        const thead = document.getElementById('preview-thead');
        const tbody = document.getElementById('preview-tbody');
        
        if (!dataArray || dataArray.length === 0) {
            section.classList.add('hidden');
            return;
        }

        section.classList.remove('hidden');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        // Headers
        const headers = Object.keys(dataArray[0]);
        const trHead = document.createElement('tr');
        headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            trHead.appendChild(th);
        });
        thead.appendChild(trHead);

        // Rows
        dataArray.forEach(row => {
            const tr = document.createElement('tr');
            headers.forEach(h => {
                const td = document.createElement('td');
                td.textContent = row[h];
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    // Modal Logic
    const btnUnassigned = document.getElementById('btn-unassigned-modal');
    const modal = document.getElementById('unassigned-modal');
    const btnCloseModal = document.getElementById('close-modal');
    const modalTbody = document.getElementById('modal-unassigned-tbody');

    btnUnassigned.addEventListener('click', () => {
        if (window.currentUnassignedStores && window.currentUnassignedStores.length > 0) {
            modalTbody.innerHTML = '';
            window.currentUnassignedStores.forEach(store => {
                const tr = document.createElement('tr');
                const tdName = document.createElement('td');
                tdName.textContent = store.tienda || 'Desconocida';
                const tdCount = document.createElement('td');
                tdCount.textContent = store.cantidad || 0;
                tr.appendChild(tdName);
                tr.appendChild(tdCount);
                modalTbody.appendChild(tr);
            });
            modal.classList.remove('hidden');
        } else {
            alert('No hay tiendas sin asignar en este momento.');
        }
    });

    btnCloseModal.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});
