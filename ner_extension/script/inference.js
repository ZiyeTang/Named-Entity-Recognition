
window.onload = function () {
  let prevText = "";
  $( document ).ready(function() {
    document.addEventListener("mouseup", function(){
        var selectedText = window.getSelection().toString();
        // console.log(selectedText);
        if (selectedText!="" && selectedText!=prevText) {
          prevText = selectedText
          $.ajax({
              url: "http://127.0.0.1:5000/",
              type: "POST",
              dataType: "text",
              data: selectedText,
              success: function(response) {
                console.log("Success");
                alert(response);
              },
              error: function(error) {
                console.log("Error:", error);
              }
          });
        } else {
          prevText = ""
        }
    });
  });
}