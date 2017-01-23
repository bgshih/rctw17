$.get("nav-jumbo.html", function(data){
  $("#nav-jumbo-placeholder").replaceWith(data);
});

$.get("footer.html", function(data){
  $("#footer").replaceWith(data);
});
