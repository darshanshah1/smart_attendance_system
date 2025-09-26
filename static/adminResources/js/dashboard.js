(function ($) {
    'use strict';
    $(function () {
        if ($("#analysis-chart").length) {
            var CurrentChartCanvas = $("#analysis-chart").get(0).getContext("2d");
            var CurrentChart = new Chart(CurrentChartCanvas, {
                type: 'bar',
                data: {
                    labels: ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
                    datasets: [{
                        label: 'Profit',
                        data: [280, 330, 370, 410, 290, 400, 309, 530, 340, 420],
                        backgroundColor: '#633e77'
                    },
                        {
                            label: 'Target',
                            data: [380, 540, 600, 480, 370, 500, 450, 590, 540, 480],
                            backgroundColor: '#e7eaed'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    layout: {
                        padding: {
                            left: 0,
                            right: 0,
                            top: 20,
                            bottom: 0
                        }
                    },
                    scales: {
                        yAxes: [{
                            display: false,
                            gridLines: {
                                drawBorder: false
                            },
                            ticks: {
                                display: false
                            }
                        }],
                        xAxes: [{
                            stacked: true,
                            ticks: {
                                beginAtZero: true,
                                fontColor: "#9fa0a2"
                            },
                            gridLines: {
                                color: "rgba(0, 0, 0, 0)",
                                display: true
                            },
                            barPercentage: 0.6
                        }]
                    },
                    legend: {
                        display: false
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }
        if ($("#analysis-chart-dark").length) {
            var CurrentChartCanvas = $("#analysis-chart-dark").get(0).getContext("2d");
            var CurrentChart = new Chart(CurrentChartCanvas, {
                type: 'bar',
                data: {
                    labels: ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
                    datasets: [{
                        label: 'Profit',
                        data: [280, 330, 370, 410, 290, 400, 309, 530, 340, 420],
                        backgroundColor: '#8110a5'
                    },
                        {
                            label: 'Target',
                            data: [380, 540, 600, 480, 370, 500, 450, 590, 540, 480],
                            backgroundColor: '#848086'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    layout: {
                        padding: {
                            left: 0,
                            right: 0,
                            top: 20,
                            bottom: 0
                        }
                    },
                    scales: {
                        yAxes: [{
                            display: false,
                            gridLines: {
                                drawBorder: false
                            },
                            ticks: {
                                display: false
                            }
                        }],
                        xAxes: [{
                            stacked: true,
                            ticks: {
                                beginAtZero: true,
                                fontColor: "#9fa0a2"
                            },
                            gridLines: {
                                color: "rgba(0, 0, 0, 0)",
                                display: true
                            },
                            barPercentage: 0.6
                        }]
                    },
                    legend: {
                        display: false
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }
        if ($("#traffic-chart").length) {
            var trafficChartData = {
                datasets: [{
                    data: [40, 30, 30],
                    backgroundColor: [
                        "#ff4979",
                        "#0acf97",
                        "#633e77"
                    ],
                    hoverBackgroundColor: [
                        "#ff4979",
                        "#0acf97",
                        "#633e77"
                    ],
                    borderColor: [
                        "#ff4979",
                        "#0acf97",
                        "#633e77"
                    ],
                    legendColor: [
                        "#ff4979",
                        "#0acf97",
                        "#633e77"
                    ]
                }],

                // These labels appear in the legend and in the tooltips when hovering different arcs
                labels: [
                    'Electronics',
                    'Clothing',
                    'Beverages',
                ]
            };
            var trafficChartOptions = {
                responsive: true,
                animation: {
                    animateScale: true,
                    animateRotate: true
                },
                legend: false,
                legendCallback: function (chart) {
                    var text = [];
                    text.push('<ul>');
                    for (var i = 0; i < trafficChartData.datasets[0].data.length; i++) {
                        text.push('<li><div class="d-flex align-items-center"><span class="legend-dots" style="background:' +
                            trafficChartData.datasets[0].legendColor[i] +
                            '"></span><span>' + trafficChartData.labels[i] + '</span></div><div>' + trafficChartData.datasets[0].data[i] + '%</div>');
                        text.push('</li>');
                    }
                    text.push('</ul>');
                    return text.join('');
                },
                cutoutPercentage: 78
            };
            var trafficChartPlugins = {
                beforeDraw: function (chart) {
                    var width = chart.chart.width,
                        height = chart.chart.height,
                        ctx = chart.chart.ctx;

                    ctx.restore();
                    var fontSize = 1.2;
                    ctx.font = fontSize + "em sans-serif";
                    ctx.textBaseline = "middle";
                    ctx.fillStyle = "#76838f";

                    var text = "42%",
                        textX = Math.round((width - ctx.measureText(text).width) / 2),
                        textY = height / 2;

                    ctx.fillText(text, textX, textY);
                    ctx.save();
                }
            }
            var trafficChartCanvas = $("#traffic-chart").get(0).getContext("2d");
            var trafficChart = new Chart(trafficChartCanvas, {
                type: 'doughnut',
                data: trafficChartData,
                options: trafficChartOptions,
                plugins: trafficChartPlugins
            });
            $("#traffic-chart-legend").html(trafficChart.generateLegend());
        }
        if ($("#traffic-chart-dark").length) {
            var trafficChartData = {
                datasets: [{
                    data: [40, 30, 30],
                    backgroundColor: [
                        "#ff4979",
                        "#0acf97",
                        "#8110a5"
                    ],
                    hoverBackgroundColor: [
                        "#ff4979",
                        "#0acf97",
                        "#8110a5"
                    ],
                    borderColor: [
                        "#ff4979",
                        "#0acf97",
                        "#8110a5"
                    ],
                    legendColor: [
                        "#ff4979",
                        "#0acf97",
                        "#8110a5"
                    ]
                }],

                // These labels appear in the legend and in the tooltips when hovering different arcs
                labels: [
                    'Electronics',
                    'Clothing',
                    'Beverages',
                ]
            };
            var trafficChartOptions = {
                responsive: true,
                animation: {
                    animateScale: true,
                    animateRotate: true
                },
                legend: false,
                legendCallback: function (chart) {
                    var text = [];
                    text.push('<ul>');
                    for (var i = 0; i < trafficChartData.datasets[0].data.length; i++) {
                        text.push('<li><div class="d-flex align-items-center"><span class="legend-dots" style="background:' +
                            trafficChartData.datasets[0].legendColor[i] +
                            '"></span><span>' + trafficChartData.labels[i] + '</span></div><div>' + trafficChartData.datasets[0].data[i] + '%</div>');
                        text.push('</li>');
                    }
                    text.push('</ul>');
                    return text.join('');
                },
                cutoutPercentage: 78
            };
            var trafficChartPlugins = {
                beforeDraw: function (chart) {
                    var width = chart.chart.width,
                        height = chart.chart.height,
                        ctx = chart.chart.ctx;

                    ctx.restore();
                    var fontSize = 1.2;
                    ctx.font = fontSize + "em sans-serif";
                    ctx.textBaseline = "middle";
                    ctx.fillStyle = "#76838f";

                    var text = "42%",
                        textX = Math.round((width - ctx.measureText(text).width) / 2),
                        textY = height / 2;

                    ctx.fillText(text, textX, textY);
                    ctx.save();
                }
            }
            var trafficChartCanvas = $("#traffic-chart-dark").get(0).getContext("2d");
            var trafficChart = new Chart(trafficChartCanvas, {
                type: 'doughnut',
                data: trafficChartData,
                options: trafficChartOptions,
                plugins: trafficChartPlugins
            });
            $("#traffic-chart-legend").html(trafficChart.generateLegend());
        }
        if ($("#statistics-chart").length) {
            var areaData = {
                labels: ["IA", "RI", "NY", "CO", "MI", "FL", "IL", "PA", "LA", "NJ", "CA", "TX", "LA", "PQ", "RF", "JG"],
                datasets: [{
                    data: [60, 40, 55, 80, 60, 60, 60, 65, 30, 60, 55, 80, 55, 40, 75, 50],
                    backgroundColor: [
                        '#633e77'
                    ],
                    borderColor: [
                        '#633e77'
                    ],
                    borderWidth: 1,
                    fill: 'origin',
                    label: "purchases"
                },
                    {
                        data: [75, 55, 70, 95, 75, 75, 75, 80, 55, 75, 70, 95, 70, 55, 90, 65],
                        backgroundColor: [
                            'rgba(255, 73, 121, 1)'
                        ],
                        borderColor: [
                            'rgba(255, 73, 121, 1)'
                        ],
                        borderWidth: 1,
                        fill: 'origin',
                        label: "services"
                    }
                ]
            };
            var areaOptions = {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    filler: {
                        propagate: false
                    }
                },
                scales: {
                    xAxes: [{
                        display: true,
                        ticks: {
                            display: true
                        },
                        gridLines: {
                            display: false,
                            drawBorder: false,
                            color: 'transparent',
                            zeroLineColor: '#eeeeee'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        ticks: {
                            display: true,
                            autoSkip: false,
                            maxRotation: 0,
                            stepSize: 20,
                            min: 0,
                            max: 100
                        },
                        gridLines: {
                            drawBorder: false
                        }
                    }]
                },
                legend: {
                    display: false
                },
                tooltips: {
                    enabled: true
                },
                elements: {
                    line: {
                        tension: 0
                    },
                    point: {
                        radius: 0
                    }
                }
            }
            var salesChartCanvas = $("#statistics-chart").get(0).getContext("2d");
            var salesChart = new Chart(salesChartCanvas, {
                type: 'line',
                data: areaData,
                options: areaOptions
            });
        }
        if ($("#statistics-chart-dark").length) {
            var areaData = {
                labels: ["IA", "RI", "NY", "CO", "MI", "FL", "IL", "PA", "LA", "NJ", "CA", "TX", "LA", "PQ", "RF", "JG"],
                datasets: [{
                    data: [60, 40, 55, 80, 60, 60, 60, 65, 30, 60, 55, 80, 55, 40, 75, 50],
                    backgroundColor: [
                        '#4e0e63'
                    ],
                    borderColor: [
                        '#4e0e63'
                    ],
                    borderWidth: 1,
                    fill: 'origin',
                    label: "purchases"
                },
                    {
                        data: [75, 55, 70, 95, 75, 75, 75, 80, 55, 75, 70, 95, 70, 55, 90, 65],
                        backgroundColor: [
                            'rgba(255, 255, 255, .1)'
                        ],
                        borderColor: [
                            'rgba(255, 255, 255, .1)'
                        ],
                        borderWidth: 1,
                        fill: 'origin',
                        label: "services"
                    }
                ]
            };
            var areaOptions = {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    filler: {
                        propagate: false
                    }
                },
                scales: {
                    xAxes: [{
                        display: true,
                        ticks: {
                            display: true
                        },
                        gridLines: {
                            display: false,
                            drawBorder: false,
                            color: 'transparent',
                            zeroLineColor: '#eeeeee'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        ticks: {
                            display: true,
                            autoSkip: false,
                            maxRotation: 0,
                            stepSize: 20,
                            min: 0,
                            max: 100
                        },
                        gridLines: {
                            drawBorder: false,
                            color: "#312d4e"
                        }
                    }]
                },
                legend: {
                    display: false
                },
                tooltips: {
                    enabled: true
                },
                elements: {
                    line: {
                        tension: 0
                    },
                    point: {
                        radius: 0
                    }
                }
            }
            var salesChartCanvas = $("#statistics-chart-dark").get(0).getContext("2d");
            var salesChart = new Chart(salesChartCanvas, {
                type: 'line',
                data: areaData,
                options: areaOptions
            });
        }
        if ($("#inline-datepicker-example").length) {
            $('#inline-datepicker-example').datepicker({
                enableOnReadonly: true,
                todayHighlight: true,
                templates: {
                    leftArrow: '<i class="mdi mdi-chevron-left"></i>',
                    rightArrow: '<i class="mdi mdi-chevron-right"></i>'
                }
            });
        }
    });
});