$("#history_button").click(function(){
	$(location).attr("href","/history/");
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
			console.log(result);
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
	$(location).attr("href",'/result/');
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
	fd.append("image",imageData, slider);
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



function tags(id){
	
}

var slider = document.getElementById("slider");
slider.onchange = function(event){
	var output = document.getElementById("output");
	output.innerHTML = slider.value;
}