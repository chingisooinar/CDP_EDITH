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

function tags(id){
	console.log(id);
}

function showEidingState(){
	$("#stateTitle").html("");
	$("#editing_button").hide();
	$("#generate_button").show();
	$("#colorize_button").show();
	$("#complete_button").show();
	$("#download_button").hide();
	$("#save_button").hide();
	$(".tag_menu").show();
}

function showFinishState(){
	$("#stateTitle").html("Final Product");
	$("#editing_button").css("display","inline-block");
	$("#generate_button").hide();
	$("#colorize_button").hide();
	$("#complete_button").hide();
	$("#download_button").show();
	$("#save_button").show();
	$(".tag_menu").hide();
}

function showHistory(){
	$("#stateTitle").html("My History");
	$("#history_button").css("display","inline-block");
}