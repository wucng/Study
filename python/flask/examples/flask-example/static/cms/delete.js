
$(function (){
    // #+id, .+class
    $("#delete").click(function (event) {
        // 是阻止按钮默认的提交表单的事件
        event.preventDefault();

        var telephone = $("#delete").attr("content");

        // 1. 要在模版的meta标签中渲染一个csrf-token
        // 2. 在ajax请求的头部中设置X-CSRFtoken
        zlajax.post({
            'url':"/cms/delete/",
            'data':{
                'telephone':telephone
            },
            'success': function (data) {
                if(data['code']==200){
                    zlalert.alertSuccessToast("恭喜！删除成功！");
                    window.location = '/cms/details/'; // 跳转到新页面
                    // telephoneE.val("");
                    // passwordE.val("");
                    // rememberE.val("");

                }else {
                    zlalert.alertInfo(data["message"]);
                }
            },
            'fail': function (error) {
                // console.log(error);
                zlalert.alertNetworkError();
            }
        });
    })
});