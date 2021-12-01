$("#edit_button").click(function(){
	var form = $("<form method='post'></form>");
	form.attr({"action":'/index/'});
	var input = $("<input type='hidden'>");
	input.attr({"name":"filename"});
	input.val($.FILENAME);
	form.append(input);
	$(document.body).append(form);
	form.submit();
})

$("#history_button").click(function(){
	$(location).attr("href","/history/");
})

$("#download_button").click(function(){
	var download = $("<a href='/media/user/"+ $.FILENAME +"' download='" + $.FILENAME + "'></a>");
	$(document.body).append(download);
	download[0].click();
})

$("#saveas_button").click(function(){
	
})