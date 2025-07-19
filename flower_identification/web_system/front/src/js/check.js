var IMAGE_URL = '';
var knowledge = '';
function Check() {

}

Check.prototype.listenUploadFileEvent = function () {
    var uploadBtn = $('#upload-btn');
    uploadBtn.change(function () {
        // uploadBtn[0]：找到第一个uploadBtn标签
        var file = uploadBtn[0].files[0];
        var formData = new FormData();
        formData.append('file', file);
        $.ajax({
            'type': 'POST',
            'url': '/upload_img/',
            'data': formData,
            // 告诉jQuery不需要对文件格式进行处理了
            'processData': false,
            // 使用默认的文件形式，不需要在添加了
            'contentType': false,
            'success': function (result) {
                if (result['code'] === 200) {
                    IMAGE_URL = result['data']['url'];
                    var img_url = result['data']['url'];
                    var imageInput = $('#img-url');
                    imageInput.val(img_url);
                    var inputImg = $('.input-image-file');
                    inputImg.attr('src', img_url)
                }
            }
        })
    })
};


Check.prototype.listenImgCheckEvent = function () {
    var changeBtn = $('#img-check');
    changeBtn.click(function (event) {
        myalert.alertInfoWithTitle('请稍等，正在识别中！')
        event.preventDefault();
        $.ajax({
            'type': 'POST',
            'url': '/check_img/',
            'data': {
                'img_url': IMAGE_URL,
            },
            'success': function (result) {
                if (result['code'] === 200) {
                    var pred_name = result['data']['pred_name'];
                    knowledge = result['data']['query']
                    console.log(knowledge)
                    var know = $('#know-ledge')
                    know.text(knowledge)
                    myalert.alertInfoWithTitle(pred_name, '识别成功!');
                } else {
                    myalert.alertInfoWithTitle(result['message'], '错误信息')
                }
            }
        })
    })
}


Check.prototype.run = function () {
    var self = this;
    self.listenUploadFileEvent();
    self.listenImgCheckEvent();
};


$(function () {
    var check = new Check();
    check.run()
});



