let x  = window.location.toString();
x = x.substring(8);
console.log(x);
fetch("http://127.0.0.1:5000/predict1", {
  method: "POST",
  body: JSON.stringify({
    "url" : x 
  }),
  headers: {
    "Content-type": "application/json; charset=UTF-8"
  }
})
  .then((response) => response.json())
  .then((json) => {console.log(json);alert(json.values);});