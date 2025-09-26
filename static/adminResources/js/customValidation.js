// _______________for add area----------------------

function addAreaType() {

    if ($('#areaName').val().trim() === '') {
        $('#areaName').focus()
        showErrorToast(' Please enter area name')
        return false;
    }
    else if ($('#areaName').val().length < 3) {
        $('#areaName').focus()
        showErrorToast(' Please enter valid area name')
        return false;
    }
    else if ($('#areaPincode').val().trim() === '') {
        $('#areaPincode').focus()
        showErrorToast(' Please enter area pincode')
        return false;
    }

    else if ($('#areaPincode').val().length != 6) {
        $('#areaPincode').focus()
        showErrorToast(' Please enter valid area pincode')
        return false;
    }
    else {
        return true;
    }
}

// _______________for add category----------------------

function addCategoryType() {


    if ($('#categoryName').val().trim() === '') {
        $('#categoryName').focus()
        showErrorToast(' Please enter category name')
        return false;
    }
    else if ($('#categoryName').val().length < 3) {
        $('#categoryName').focus()
        showErrorToast(' Please enter valid category name')
        return false;
    }
    else if ($('#categoryDescription').val().trim() === '') {
        $('#categoryDescription').focus()
        showErrorToast(' Please enter category description')
        return false;
    }

    else if ($('#categoryDescription').val().length < 6) {
        $('#categoryDescription').focus()
        showErrorToast(' Please enter valid category description')
        return false;
    }
    else {
        return true;
    }
}

// _______________for add subcategory----------------------

function addSubcategoryType() {

    if ($('#subCategoryCategoryId').val().trim() === 'none') {
        $('#subCategoryCategoryId').focus()
        showErrorToast('Please select category')
        return false;
    }

    else if ($('#subCategoryName').val().trim() === '') {
        $('#subCategoryName').focus()
        showErrorToast(' Please enter subcategory name')
        return false;
    }
    else if ($('#subCategoryName').val().length < 3) {
        $('#subCategoryName').focus()
        showErrorToast(' Please enter valid subcategory name')
        return false;
    }
    else if ($('#subCategoryDescription').val().trim() === '') {
        $('#subCategoryDescription').focus()
        showErrorToast(' Please enter subcategory description')
        return false;
    }

    else if ($('#subCategoryDescription').val().length < 6) {
        $('#subCategoryDescription').focus()
        showErrorToast(' Please enter valid subcategory description')
        return false;
    }
    else {
        return true;
    }
}

// _______________for add product----------------------

function addProductType() {

    if ($('#productCategoryId').val().trim() === 'none') {
        $('#productCategoryId').focus()
        showErrorToast('Please select category')
        return false;
    }

    else if ($('#productName').val().trim() === '') {
        $('#productName').focus()
        showErrorToast(' Please enter product name')
        return false;
    }
    else if ($('#productName').val().length < 3) {
        $('#productName').focus()
        showErrorToast(' Please enter valid product name')
        return false;
    }
    else if ($('#productDescription').val().trim() === '') {
        $('#productDescription').focus()
        showErrorToast(' Please enter product description')
        return false;
    }

    else if ($('#productDescription').val().length < 6) {
        $('#productDescription').focus()
        showErrorToast(' Please enter valid product description')
        return false;
    }

    else if ($('#productQuantity').val().trim() === '') {
        $('#productQuantity').focus()
        showErrorToast(' Please enter product quantity')
        return false;
    }

    else if ($('#productPrice').val().trim() === '') {
        $('#productPrice').focus()
        showErrorToast(' Please enter product price')
        return false;
    }
    else if ($('#productImage').val().trim() === '') {
        $('#productImage').focus()
        showErrorToast(' Please enter product Image')
        return false;
    }

    else {
        return true;
    }
}


// _______________for reply ----------------------

function addReplyType() {
    if ($('#complainReplyDescription').val().trim() === '') {
        $('#complainReplyDescription').focus()
        showErrorToast(' Please enter reply description')
        return false;
    }
    else if ($('#complainReplyDescription').val().length < 5) {
        $('#complainReplyDescription').focus()
        showErrorToast(' Please enter valid reply description')
        return false;
    }

    else {
        return true;
    }
}


// _______________for complain ----------------------

function addComplainType() {
    if ($('#ComplainSubject').val().trim() === '') {
        $('#ComplainSubject').focus();
        showErrorToast(' Please enter complain subject');
        return false;
    }
    else if ($('#ComplainSubject').val().length < 5) {
        $('#ComplainSubject').focus();
        showErrorToast(' Please enter valid complain subject ');
        return false;
    }
    else if ($('#ComplainDescription').val().trim() === '') {
        $('#ComplainDescription').focus();
        showErrorToast(' Please enter complain description');
        return false;
    }

    else if ($('#ComplainDescription').val().length < 8) {
        $('#ComplainDescription').focus();
        showErrorToast(' Please enter valid complain description');
        return false;
    }

    else {
        return true;
    }
}


// _______________for feedback ----------------------

function addFeedbackType() {
    if ($('#feedbackDescription').val().trim() === '') {
        $('#feedbackDescription').focus()
        showErrorToast(' Please enter Feedback Description ')
        return false;
    }
    else if ($('#feedbackDescription').val().length < 5) {
        $('#feedbackDescription').focus()
        showErrorToast(' Please enter valid Feedback Description  ')
        return false;
    }

    else {
        return true;
    }
}


// _______________for register ----------------------

function addRegisterType() {
    if ($('#userFirstname').val().trim() === '') {
        $('#userFirstname').focus()
        showErrorToast(' Please enter your firstname ')
        return false;
    }

    else if ($('#userFirstname').val().length < 3) {
        $('#userFirstname').focus()
        showErrorToast(' Please enter valid firstname ')
        return false;
    }


    else if ($('#userLastname').val().trim() === '') {
        $('#userLastname').focus()
        showErrorToast(' Please enter your lastname')
        return false;
    }

    else if ($('#userLastname').val().length < 3) {
        $('#userLastname').focus()
        showErrorToast(' Please enter valid lastname ')
        return false;
    }


    else if ($('#loginUsername').val().trim() === '') {
        $('#loginUsername').focus()
        showErrorToast(' Please enter your username')
        return false;
    }

    else if ($('#userCity').val().trim() === 'none') {
        $('#userCity').focus()
        showErrorToast('Please select city')
        return false;
    }

    else if ($('#userArea').val().trim() === 'none') {
        $('#userArea').focus()
        showErrorToast('Please select area')
        return false;
    }

    else if ($('#userAddress').val().trim() === '') {
        $('#userAddress').focus()
        showErrorToast(' Please enter your address')
        return false;
    }

    else if ($('#userAddress').val().length < 8) {
        $('#userAddress').focus()
        showErrorToast(' Please enter valid address ')
        return false;
    }

    else {
        return true;
    }
}

// _______________for login ----------------------

function addLoginType() {
    if ($('#loginUsername').val().trim() === '') {
        $('#loginUsername').focus()
        showErrorToast(' Please enter login username ')
        return false;
    }
    else if ($('#loginPassword').val().trim() === '') {
        $('#loginPassword').focus()
        showErrorToast(' Please enter login password')
        return false;
    }

    else {
        return true;
    }
}