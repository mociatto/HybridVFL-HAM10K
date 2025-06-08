const socket = io();

let dataChangeDetected = false;
let previousDataLengths = {};
let timerInterval;
let trainingInProgress = false;

// Mode selection functionality
document.addEventListener('DOMContentLoaded', function() {
    const vanillaBtn = document.getElementById('vanilla-btn');
    const fairBtn = document.getElementById('fair-btn');
    
    // Initialize with "Choose Model Mode..." status
    updateStatus('Choose Model Mode...');
    
    // Button click handlers
    vanillaBtn.addEventListener('click', function() {
        selectMode('VanillaFL', false);
    });
    
    fairBtn.addEventListener('click', function() {
        selectMode('SecureVFL', true);
    });
});

function selectMode(modeName, secureVFLValue) {
    const vanillaBtn = document.getElementById('vanilla-btn');
    const fairBtn = document.getElementById('fair-btn');
    
    // Update button states
    vanillaBtn.classList.remove('selected');
    fairBtn.classList.remove('selected');
    
    if (modeName === 'VanillaFL') {
        vanillaBtn.classList.add('selected');
    } else {
        fairBtn.classList.add('selected');
    }
    
    // If training is in progress, confirm switch
    if (trainingInProgress) {
        if (confirm(`Training is currently in progress. Do you want to stop current training and switch to ${modeName}?`)) {
            // Stop current training and start new mode
            switchMode(modeName, secureVFLValue);
        } else {
            // Revert button selection
            if (modeName === 'VanillaFL') {
                vanillaBtn.classList.remove('selected');
                fairBtn.classList.add('selected');
            } else {
                fairBtn.classList.remove('selected');
                vanillaBtn.classList.add('selected');
            }
        }
    } else {
        // Start training with selected mode
        switchMode(modeName, secureVFLValue);
    }
}

function switchMode(modeName, secureVFLValue) {
    // Disable buttons during switch
    disableButtons();
    
    // Update status
    updateStatus(`Switching to ${modeName} mode...`);
    
    // Send mode switch request to backend
    socket.emit('switch_mode', {
        mode: modeName,
        secureVFL: secureVFLValue
    });
}

function disableButtons() {
    const vanillaBtn = document.getElementById('vanilla-btn');
    const fairBtn = document.getElementById('fair-btn');
    
    vanillaBtn.classList.add('disabled');
    fairBtn.classList.add('disabled');
    vanillaBtn.disabled = true;
    fairBtn.disabled = true;
}

function enableButtons() {
    const vanillaBtn = document.getElementById('vanilla-btn');
    const fairBtn = document.getElementById('fair-btn');
    
    vanillaBtn.classList.remove('disabled');
    fairBtn.classList.remove('disabled');
    vanillaBtn.disabled = false;
    fairBtn.disabled = false;
}

function updateStatus(status) {
    const statusElement = document.getElementById('current-status');
    if (statusElement) {
        statusElement.textContent = status;
    }
}

// Listen for mode switch responses
socket.on('mode_switched', function(data) {
    console.log('Mode switched successfully:', data);
    enableButtons();
    trainingInProgress = true;
});

socket.on('mode_switch_error', function(data) {
    console.error('Mode switch error:', data);
    updateStatus('Error switching mode. Please try again.');
    enableButtons();
});

socket.on('training_completed', function(data) {
    trainingInProgress = false;
    enableButtons();
});

socket.on('training_stopped', function(data) {
    trainingInProgress = false;
    enableButtons();
    updateStatus('Training stopped. Choose a mode to restart.');
});

function plotLineChart(elementId, data, color, epochs_per_round, rounds_total) {
    if (!data || data.length === 0) {
        // Show empty placeholder
        Plotly.newPlot(elementId, [], {
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            font: { color: '#fff', family: 'Montserrat', size: 12 },
            xaxis: { 
                showgrid: false, 
                zeroline: false, 
                showline: false,
                color: '#666'
            },
            yaxis: { 
                showgrid: false, 
                zeroline: false, 
                showline: false,
                color: '#666'
            },
            margin: { l: 40, r: 20, t: 20, b: 40 }
        }, {
            displayModeBar: false,
            responsive: true
        });
        return;
    }

    // Create continuous x-axis values (1,2,3,4,5,6,7,8...)
    const xValues = [];
    const xLabels = [];
    const totalEpochs = data.length;
    
    for (let i = 0; i < totalEpochs; i++) {
        const continuousEpoch = i + 1; // Continuous numbering
        const epochInRound = (i % epochs_per_round) + 1; // Restarting labels
        xValues.push(continuousEpoch);
        xLabels.push(epochInRound.toString());
    }
    
    // Add one more epoch for next value preview
    const nextContinuousEpoch = totalEpochs + 1;
    const nextEpochLabel = ((totalEpochs % epochs_per_round) + 1).toString();
    xValues.push(nextContinuousEpoch);
    xLabels.push(nextEpochLabel);
    
    // Check if data has changed
    const currentLength = data.length;
    if (previousDataLengths[elementId] !== currentLength) {
        dataChangeDetected = true;
        previousDataLengths[elementId] = currentLength;
    }

    // Create data array with null for the next epoch placeholder
    const yValues = [...data.filter(val => val !== null && val !== undefined), null];

    const trace = {
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: 'lines+markers',
        line: { 
            color: color, 
            width: 3,
            shape: 'spline',
            smoothing: 1.3
        },
        marker: { 
            color: color, 
            size: 6,
            line: { color: '#fff', width: 1 }
        },
        fill: 'tonexty',
        fillcolor: color + '20',
        connectgaps: false
    };

    const layout = {
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        font: { 
            color: '#aaa', 
            family: 'Montserrat', 
            size: 11,
            weight: 500
        },
        xaxis: {
            title: 'EPOCH',
            titlefont: { size: 10, color: '#666' },
            showgrid: true,
            gridcolor: '#3a3d42',
            gridwidth: 1,
            zeroline: false,
            showline: true,
            linecolor: '#3a3d42',
            linewidth: 1,
            color: '#aaa',
            tickmode: 'array',
            tickvals: xValues,
            ticktext: xLabels,
            range: [0.5, nextContinuousEpoch + 0.5]
        },
        yaxis: {
            showgrid: true,
            gridcolor: '#3a3d42',
            gridwidth: 1,
            zeroline: false,
            showline: true,
            linecolor: '#3a3d42',
            linewidth: 1,
            color: '#aaa',
            rangemode: 'tozero',
            tickformat: elementId.includes('accuracy') ? '.0%' : undefined
        },
        margin: { l: 50, r: 20, t: 20, b: 50 },
        showlegend: false,
        hovermode: 'x unified',
        hoverlabel: {
            bgcolor: '#2d3035',
            bordercolor: color,
            font: { color: '#fff', size: 12 }
        }
    };

    Plotly.newPlot(elementId, [trace], layout, {
        displayModeBar: false,
        responsive: true
    }).then(() => {
        // Auto-scale y-axis if data changed
        if (dataChangeDetected) {
            const plotDiv = document.getElementById(elementId);
            if (plotDiv && plotDiv.data && plotDiv.data[0] && plotDiv.data[0].y.length > 0) {
                const yValues = plotDiv.data[0].y.filter(val => val !== null && val !== undefined);
                if (yValues.length > 0) {
                    const yMin = Math.min(...yValues);
                    const yMax = Math.max(...yValues);
                    const yRange = yMax - yMin;
                    const yPadding = yRange * 0.1;
                    
                    let newYMin = Math.max(0, yMin - yPadding);
                    let newYMax = yMax + yPadding;
                    
                    // For accuracy charts, ensure range is 0-1 if values are in that range
                    if (elementId.includes('accuracy') && yMax <= 1) {
                        newYMin = 0;
                        newYMax = Math.max(1, newYMax);
                    }
                    
                    Plotly.relayout(elementId, {
                        'yaxis.range': [newYMin, newYMax]
                    });
                }
            }
        }
    });
}

function updateTimer(startTime) {
    if (!startTime) return;
    
    const currentTime = new Date().getTime();
    const startTimeMs = new Date(startTime * 1000).getTime();
    const elapsedSeconds = Math.floor((currentTime - startTimeMs) / 1000);
    
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const timerElement = document.getElementById('running-time');
    if (timerElement) {
        timerElement.textContent = timeString;
    }
}

function stopTimer(finalTime) {
    const timerElement = document.getElementById('running-time');
    if (timerElement && finalTime) {
        timerElement.textContent = finalTime;
    }
}

socket.on('metrics_update', function(data) {
    
    // Reset data change detection flag
    dataChangeDetected = false;
    
    // Track training progress based on status
    if (data.current_status) {
        const status = data.current_status.toLowerCase();
        if (status.includes('training') || status.includes('evaluating') || status.includes('calculating')) {
            trainingInProgress = true;
        } else if (status.includes('completed') || status.includes('choose model mode')) {
            trainingInProgress = false;
            enableButtons();
        }
    }
    
    // Track training completion
    if (data.training_completed) {
        trainingInProgress = false;
        enableButtons();
    }
    
    // ENHANCED: Handle mode switching status
    if (data.mode_switching) {
        console.log(`Switching to ${data.switching_to_mode}. Previous results preserved.`);
        if (data.previous_mode_results && data.previous_mode_results.mode) {
            console.log(`Previous ${data.previous_mode_results.mode} results:`, data.previous_mode_results);
        }
    }
    
    // Update basic metrics
    if (data.batch_size !== undefined) {
        document.getElementById('batch-size').textContent = data.batch_size.toString();
    }
    
    if (data.round !== undefined && data.rounds_total !== undefined) {
        document.getElementById('current-round').textContent = `${data.round} | ${data.rounds_total}`;
    }
    
    if (data.epochs_per_round !== undefined) {
        document.getElementById('epochs-per-round').textContent = data.epochs_per_round.toString();
    }
    
    if (data.feature_class !== undefined) {
        document.getElementById('feature-classes').textContent = data.feature_class.toString();
    }
    
    if (data.current_status !== undefined) {
        document.getElementById('current-status').textContent = data.current_status.toUpperCase();
    }
    
    // Update timer with preserved final time handling
    if (data.start_time && !data.training_completed && !data.final_running_time) {
        clearInterval(timerInterval);
        timerInterval = setInterval(() => updateTimer(data.start_time), 1000);
        updateTimer(data.start_time);
    } else if (data.training_completed && data.final_running_time) {
        clearInterval(timerInterval);
        stopTimer(data.final_running_time);
    } else if (data.final_running_time) {
        // FIXED: Show preserved final time from previous training
        clearInterval(timerInterval);
        document.getElementById('running-time').textContent = data.final_running_time;
    } else if (data.running_time) {
        document.getElementById('running-time').textContent = data.running_time;
    }
    
    // Update plots with improved colors matching the image
    if (data.train_acc_history && Array.isArray(data.train_acc_history)) {
        plotLineChart('train-accuracy-chart', data.train_acc_history, '#4caf50', data.epochs_per_round, data.rounds_total);
    }
    if (data.val_acc_history && Array.isArray(data.val_acc_history)) {
        plotLineChart('val-accuracy-chart', data.val_acc_history, '#2196f3', data.epochs_per_round, data.rounds_total);
    }
    if (data.train_loss_history && Array.isArray(data.train_loss_history)) {
        plotLineChart('train-loss-chart', data.train_loss_history, '#ff9800', data.epochs_per_round, data.rounds_total);
    }
    if (data.gender_acc_history && Array.isArray(data.gender_acc_history)) {
        plotLineChart('gender-accuracy-chart', data.gender_acc_history, '#ffc107', data.epochs_per_round, data.rounds_total);
    }
    if (data.age_acc_history && Array.isArray(data.age_acc_history)) {
        plotLineChart('age-accuracy-chart', data.age_acc_history, '#9e9e9e', data.epochs_per_round, data.rounds_total);
    }
    if (data.adv_loss_history && Array.isArray(data.adv_loss_history)) {
        plotLineChart('adv-loss-chart', data.adv_loss_history, '#f44336', data.epochs_per_round, data.rounds_total);
    }

    // ENHANCED: Update final performance metrics with better preservation
    const testAccuracy = (typeof data.test_accuracy === 'number' && data.test_accuracy > 0) ? 
        (data.test_accuracy * 100).toFixed(0) + '%' : '-';
    const f1Score = (typeof data.f1_score === 'number' && data.f1_score > 0) ? 
        data.f1_score.toFixed(3) : '-';
    
    // FIXED: Always display leakage values if available (including 0 values for privacy assessment)
    const leakGender = (typeof data.leak_gender_fused === 'number') ? 
        (data.leak_gender_fused * 100).toFixed(0) + '%' : '-';
    const leakAge = (typeof data.leak_age_fused === 'number') ? 
        (data.leak_age_fused * 100).toFixed(0) + '%' : '-';
    
    // Update DOM elements
    document.getElementById('test-accuracy-score').textContent = testAccuracy;
    document.getElementById('f1-score').textContent = f1Score;
    document.getElementById('leak-gender-score').textContent = leakGender;
    document.getElementById('leak-age-score').textContent = leakAge;
    
    // ENHANCED: Add visual indicators for mode comparison
    if (data.previous_mode_results && data.previous_mode_results.mode && data.mode_switching) {
        // Add comparison tooltips or indicators (can be expanded later)
        console.log('Mode comparison available:', data.previous_mode_results);
        
        // Example: Add visual indicators showing improvement/decline
        addModeComparisonIndicators(data.previous_mode_results, {
            test_accuracy: data.test_accuracy,
            f1_score: data.f1_score,
            leak_gender: data.leak_gender_fused,
            leak_age: data.leak_age_fused
        });
    }
});

function addModeComparisonIndicators(previousResults, currentResults) {
    /**
     * Add visual indicators showing performance changes between modes
     * This function can be expanded to show detailed comparisons
     */
    if (!previousResults || !currentResults) return;
    
    // For now, just log the comparison (can be enhanced with visual indicators)
    if (currentResults.test_accuracy > 0 && previousResults.test_accuracy > 0) {
        const accuracyChange = currentResults.test_accuracy - previousResults.test_accuracy;
        const privacyChange = (currentResults.leak_gender + currentResults.leak_age) - 
                             (previousResults.leak_gender + previousResults.leak_age);
        
        console.log(`Performance comparison vs ${previousResults.mode}:`);
        console.log(`  Accuracy: ${accuracyChange >= 0 ? '+' : ''}${(accuracyChange * 100).toFixed(1)}pp`);
        console.log(`  Privacy (lower is better): ${privacyChange >= 0 ? '+' : ''}${(privacyChange * 100).toFixed(1)}pp`);
    }
}

socket.on('connect', function() {});

socket.on('disconnect', function() {clearInterval(timerInterval);}); 
