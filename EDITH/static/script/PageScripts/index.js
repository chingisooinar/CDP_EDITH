$("#history_button").click(function(){
	
})

$("#generate_button").click(function(){
	$.ajax({
		type: "POST",
		url: "/api/sketch_api/",
		data: "",
		success: function(result){
			console.log(result);
			$("#sketch_canvas").wPaint("image", "data:image/png;base64,"+result);
		}
	})
})

$("#colorize_button").click(function(){
	
})

$("#complete_button").click(function(){
	
})