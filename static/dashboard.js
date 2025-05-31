// static/dashboard.js

let startTime = null;
let timerInterval = null;
let trainingCompleted = false;

// Track data lengths to detect new data
let dataLengths = {};

function updateTimer() {
    if (!startTime || trainingCompleted) return;
    const now = Date.now() / 1000;
    const elapsed = Math.floor(now - startTime);
    const min = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const sec = String(elapsed % 60).padStart(2, '0');
    document.getElementById('running-time').textContent = `${min}:${sec}`;
}

function stopTimer(finalTime) {
    trainingCompleted = true;
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    if (finalTime) {
        document.getElementById('running-time').textContent = finalTime;
    }
}

function plotLineChart(divId, arr, color, epochsPerRound, totalRounds, yRange=null) {
    // Filter out None/null values but keep track of original positions
    let originalData = arr && Array.isArray(arr) ? arr : [];
    let y = [];
    let validIndices = [];
    
    for (let i = 0; i < originalData.length; i++) {
        if (originalData[i] !== null && originalData[i] !== undefined && !isNaN(originalData[i])) {
            y.push(originalData[i]);
            validIndices.push(i + 1); // 1-indexed for display
        }
    }
    
    // Check if we have new data
    const hasNewData = !dataLengths[divId] || dataLengths[divId] !== y.length;
    dataLengths[divId] = y.length;
    
    // If no valid data, show empty chart
    if (y.length === 0) {
        y = [null];
        validIndices = [1];
    }
    
    // Add next epoch placeholder
    let x = validIndices.slice(); // Copy validIndices
    let nextEpoch = (originalData.length > 0) ? originalData.length + 1 : 1;
    x.push(nextEpoch);
    y = y.concat([null]);
    
    // Create tick labels to show round boundaries
    let tickvals = [];
    let ticktext = [];
    for (let i = 0; i < x.length; i++) {
        let epochNum = x[i];
        let roundNum = Math.ceil(epochNum / epochsPerRound);
        let epochInRound = ((epochNum - 1) % epochsPerRound) + 1;
        
        if (epochInRound === 1 || epochInRound === epochsPerRound || i === x.length - 1) {
            tickvals.push(epochNum);
            ticktext.push(`R${roundNum}E${epochInRound}`);
        }
    }
    
    // Determine y-axis range based on chart type
    let yAxisConfig;
    if (divId.includes('accuracy')) {
        // For accuracy charts, start with autorange but prefer 0-1 bounds
        yAxisConfig = {autorange: true, rangemode: 'tozero'};
    } else if (yRange) {
        yAxisConfig = {range: yRange, autorange: false};
    } else {
        // For other charts, use autorange with zero baseline
        yAxisConfig = {autorange: true, rangemode: 'tozero'};
    }
    
    Plotly.newPlot(divId, [{
        x: x,
        y: y,
        type: 'scatter', 
        mode: 'lines+markers', 
        line: {color: color}
    }], {
        margin: {t: 24},
        yaxis: yAxisConfig,
        xaxis: {
            title: 'Epoch', 
            range: [1, Math.max(2, Math.max(...x))], 
            dtick: 1,
            tickvals: tickvals,
            ticktext: ticktext
        }
    }).then(function() {
        // Automatically trigger y-axis autoscale only when new data is added
        if (hasNewData) {
            if (divId.includes('accuracy')) {
                // For accuracy charts, auto-fit but prefer good bounds
                Plotly.relayout(divId, {
                    'yaxis.autorange': true
                });
            } else {
                // For loss charts, fully autoscale y-axis
                Plotly.relayout(divId, {
                    'yaxis.autorange': true
                });
            }
        }
    });
}

const socket = io();

socket.on('metrics_update', function(data) {
    // Live timer logic
    if (data.start_time && (!startTime || startTime !== data.start_time)) {
        startTime = data.start_time;
        trainingCompleted = false;
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(updateTimer, 1000);
        updateTimer();
    }

    // Handle training completion
    if (data.training_completed) {
        stopTimer(data.final_running_time);
    }

    // Update status
    if (data.current_status) {
        document.getElementById('current-status').textContent = data.current_status;
    }

    document.getElementById('batch-size').textContent = data.batch_size || '-';
    // Rounds completed: show as (current_round | total_rounds), default current_round to 0
    const currentRound = (typeof data.round === 'number' ? data.round : 0);
    document.getElementById('rounds-completed').textContent = `${currentRound} | ${data.rounds_total || '-'}`;
    document.getElementById('epochs-per-round').textContent = data.epochs_per_round || '-';
    document.getElementById('sample-count').textContent = data.sample_count || '-';
    document.getElementById('feature-class').textContent = data.feature_class || '-';
    // running-time is handled by timer or final time

    // Plot charts - ensure we have valid data but handle None values
    if (data.train_acc_history && Array.isArray(data.train_acc_history)) {
        plotLineChart('train-accuracy-chart', data.train_acc_history, '#4caf50', data.epochs_per_round, data.rounds_total);
    }
    if (data.val_acc_history && Array.isArray(data.val_acc_history)) {
        plotLineChart('val-accuracy-chart', data.val_acc_history, '#4fc3f7', data.epochs_per_round, data.rounds_total);
    }
    if (data.train_loss_history && Array.isArray(data.train_loss_history)) {
        plotLineChart('train-loss-chart', data.train_loss_history, '#ffb300', data.epochs_per_round, data.rounds_total);
    }
    if (data.gender_acc_history && Array.isArray(data.gender_acc_history)) {
        plotLineChart('gender-accuracy-chart', data.gender_acc_history, '#ab47bc', data.epochs_per_round, data.rounds_total);
    }
    if (data.age_acc_history && Array.isArray(data.age_acc_history)) {
        plotLineChart('age-accuracy-chart', data.age_acc_history, '#29b6f6', data.epochs_per_round, data.rounds_total);
    }
    if (data.adv_loss_history && Array.isArray(data.adv_loss_history)) {
        plotLineChart('adv-loss-chart', data.adv_loss_history, '#ef5350', data.epochs_per_round, data.rounds_total);
    }

    // Update leakage scores - simplified to 2 boxes since image/tabular are identical
    const leakGender = (typeof data.leak_gender_image === 'number') ? data.leak_gender_image.toFixed(3) : '0.000';
    const leakAge = (typeof data.leak_age_image === 'number') ? data.leak_age_image.toFixed(3) : '0.000';
    
    // Update performance scores
    const testAccuracy = (typeof data.test_accuracy === 'number') ? (data.test_accuracy * 100).toFixed(1) + '%' : '0.0%';
    const f1Score = (typeof data.f1_score === 'number') ? data.f1_score.toFixed(3) : '0.000';
    
    document.getElementById('test-accuracy-score').textContent = testAccuracy;
    document.getElementById('f1-score').textContent = f1Score;
    document.getElementById('leak-gender-score').textContent = leakGender;
    document.getElementById('leak-age-score').textContent = leakAge;
});
