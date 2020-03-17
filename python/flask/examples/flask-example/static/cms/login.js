
$(function (){
    // #+id, .+class
    $("#submit-btn").click(function (event) {
        // 是阻止按钮默认的提交表单的事件
        event.preventDefault();
        var telephoneE = $("input[name=telephone]");
        var passwordE = $("input[name=password]");
        var rememberE = $("input[name=remember]");

        var telephone = telephoneE.val();
        var password = passwordE.val();
        var remember = rememberE.val();

        // 1. 要在模版的meta标签中渲染一个csrf-token
        // 2. 在ajax请求的头部中设置X-CSRFtoken
        zlajax.post({
            'url':"/cms/login/",
            'data':{
                'telephone':telephone,
                'password':password,
                'remember':remember
            },
            'success': function (data) {
                if(data['code']==200){
                    zlalert.alertSuccessToast("恭喜！登录成功！");
                    window.location = '/cms/'; // 跳转到新页面
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