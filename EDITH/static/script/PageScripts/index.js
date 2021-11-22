$("#history_button").click(function(){
	$(location).attr("href","/history/");
})

$("#generate_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/sketch_api/',
		data: fd,
		type: "POST",
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
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

$("#complete_button").click(function(){
	
})

$("#convert_sketch_button").click(function(){
	var imageData = $("#sketch_canvas").wPaint("image");
	var fd = new FormData();
	fd.append("image",imageData);
	$.ajax({
		url: '/api/convert_to_sketch_api/',
		data: fd,
		type: "POST",
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
		success: function(result){
			$("#sketch_canvas").wPaint("image","data:image/png;base64,"+result);
		}
	})
})

function tags(id){
	
}