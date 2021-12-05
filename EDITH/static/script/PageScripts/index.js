$("#history_button").click(function(){
	$(location).attr("href","/history/");
})

$("#inpainting_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	
	$.ajax({
		type: "POST",
		url: '/api/inpainting_api/',
		data: fd,
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#generate_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	
	$.ajax({
		type: "POST",
		url: '/api/sketch_api/',
		data: fd,
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#colorize_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/colorize_api/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#complete_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/complete/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			var form = $("<form method='post'></form>");
			form.attr({"action":'/result/'});
			var input = $("<input type='hidden'>");
			input.attr({"name":"filename"});
			input.val(result);
			form.append(input);
			$(document.body).append(form);
			form.submit();
		}
	})
	
})

$("#convert_sketch_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/convert_to_sketch_api/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#convert_bw_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/convert_to_bw_api/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#convert_bw_from_sketch_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var slider = document.getElementById("slider").value;
	var fd = new FormData();
	fd.append("image",imageData);
	fd.append("slider",slider);
	$.ajax({
		url: '/api/convert_edge_to_bw_api/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#load_self_img_button").click(function(){
	$("#load_image").click();
})

$("#load_image").change(function(){
	var img = document.getElementById("load_image").files[0];
	console.log(1);
	var fd = new FormData();
	fd.append("image",img);
	$.ajax({
		url: 'api/upload_resize_api/',
		data: fd,
		type: "POST",
		processData: false,
		contentType: false,
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#slider").change(function(){
	$("#output").html($("#slider").val());
})