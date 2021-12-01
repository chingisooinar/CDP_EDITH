$("#edit_button").click(function(){
	$(location).attr("href","/index/");
})

function showMenu(id){
	$("#"+id).show();
}

function hideMenu(id){
	$("#"+id).hide();
}

function edit(filename){
	var form = $("<form method='post'></form>");
	form.attr({"action":'/index/'});
	var input = $("<input type='hidden'>");
	input.attr({"name":"filename"});
	input.val(filename);
	form.append(input);
	$(document.body).append(form);
	form.submit();
}

function download(filename){
	var download = $("<a href='/media/user/"+ filename +"' download='" + filename + "'></a>");
	$(document.body).append(download);
	download[0].click();
}

function del(filename){
	$.ajax({
		url: '/api/deleteHistory/',
		type: "POST",
		data: "filename="+filename,
		success: function(result){
			if(result==1){
				window.location.reload();
			}else{
				alert("Something wrong!");
			}
		}
	})
}