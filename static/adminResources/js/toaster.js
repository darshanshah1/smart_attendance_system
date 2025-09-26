function showErrorToast(message) {
    $.toast({
        heading: 'Warning',
        text: message,
        showHideTransition: 'slide',
        icon: 'error',
        loaderBg: '#f2a654',
        position: 'top-right'
    })
}