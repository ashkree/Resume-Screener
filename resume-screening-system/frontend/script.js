const BASE_URL = "https://irsas.onrender.com/"; // replace with your backend Render URL

function login() {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  const form = new FormData();
  form.append("email", email);
  form.append("password", password);

  fetch(BASE_URL + "/login", { method: "POST", body: form })
    .then(res => res.json())
    .then(data => {
      localStorage.setItem("token", data.token);
      localStorage.setItem("role", data.role);
      if (data.role === "applicant") location.href = "applicant.html";
      else location.href = "hr.html";
    });
}

function uploadCV() {
  const file = document.getElementById("cv").files[0];
  const form = new FormData();
  form.append("file", file);
  form.append("token", localStorage.getItem("token"));

  fetch(BASE_URL + "/upload", { method: "POST", body: form })
    .then(res => res.json())
    .then(data => document.getElementById("result").innerText = data.review);
}

function getReview() {
  fetch(`${BASE_URL}/review?token=${localStorage.getItem("token")}`)
    .then(res => res.json())
    .then(data => document.getElementById("result").innerText = data.review);
}

if (window.location.pathname.includes("hr.html")) {
  fetch(`${BASE_URL}/applicants?token=${localStorage.getItem("token")}`)
    .then(res => res.json())
    .then(applicants => {
      const ul = document.getElementById("applicants");
      applicants.forEach(email => {
        const li = document.createElement("li");
        li.textContent = email;
        li.onclick = () => {
          fetch(`${BASE_URL}/applicant_review?email=${email}&token=${localStorage.getItem("token")}`)
            .then(res => res.json())
            .then(data => document.getElementById("hr-review").innerText = data.review);
        };
        ul.appendChild(li);
      });
    });
}
