
$(function (){
    // #+id, .+class
    $("#submit-btn").click(function (event) {
        // 是阻止按钮默认的提交表单的事件
        event.preventDefault();
        var telephone = $("input[name=telephone]").val();
        var email = $("input[name=email]").val();
        var username = $("input[name=username]").val();

        var password = $("input[name=password]").val();
        var password2 = $("input[name=password2]").val();


        // 1. 要在模版的meta标签中渲染一个csrf-token
        // 2. 在ajax请求的头部中设置X-CSRFtoken
        zlajax.post({
            'url':"/cms/logup/",
            'data':{
                'telephone':telephone,
                'email':email,
                'username':username,
                'password':password,
                'password2':password2
            },
            'success': function (data) {
                if(data['code']==200){
                    zlalert.alertSuccessToast("恭喜！注册成功！");
                    window.location = '/cms/login/'; // 跳转到新页面

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