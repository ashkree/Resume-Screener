const BASE_URL = "https://irsas.onrender.com"; // no trailing slash

function login() {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  if (!email || !password) {
    alert("Please enter email and password");
    return;
  }

  const form = new FormData();
  form.append("email", email);
  form.append("password", password);

  fetch(BASE_URL + "/login", { method: "POST", body: form })
    .then(res => {
      if (!res.ok) {
        alert("Login failed: " + res.statusText);
        throw new Error("Login failed");
      }
      return res.json();
    })
    .then(data => {
      localStorage.setItem("token", data.token);
      localStorage.setItem("role", data.role);
      if (data.role === "applicant") location.href = "applicant.html";
      else location.href = "hr.html";
    })
    .catch(err => {
      console.error("Login error:", err);
    });
}

function uploadCV() {
  const file = document.getElementById("cv").files[0];
  if (!file) {
    alert("Please select a file to upload.");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("token", localStorage.getItem("token"));

  fetch(BASE_URL + "/upload", { method: "POST", body: form })
    .then(res => {
      if (!res.ok) {
        alert("Failed to upload CV: " + res.statusText);
        throw new Error("Upload failed");
      }
      return res.json();
    })
    .then(data => {
      document.getElementById("result").innerText = data.review;
    })
    .catch(err => {
      console.error("Upload CV error:", err);
    });
}

function getReview() {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("You need to login first.");
    return;
  }

  fetch(`${BASE_URL}/review?token=${token}`)
    .then(res => {
      if (!res.ok) {
        alert("Failed to get review: " + res.statusText);
        throw new Error("Failed to get review");
      }
      return res.json();
    })
    .then(data => {
      document.getElementById("result").innerText = data.review;
    })
    .catch(err => {
      console.error("Get review error:", err);
    });
}

if (window.location.pathname.includes("hr.html")) {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("You need to login first.");
    location.href = "index.html";
  } else {
    fetch(`${BASE_URL}/applicants?token=${token}`)
      .then(res => {
        if (!res.ok) {
          alert("Failed to fetch applicants: " + res.statusText);
          throw new Error("Failed to fetch applicants");
        }
        return res.json();
      })
      .then(applicants => {
        const ul = document.getElementById("applicants");
        ul.innerHTML = ""; // clear existing list
        applicants.forEach(email => {
          const li = document.createElement("li");
          li.textContent = email;
          li.onclick = () => {
            fetch(`${BASE_URL}/applicant_review?email=${encodeURIComponent(email)}&token=${token}`)
              .then(res => {
                if (!res.ok) {
                  alert("Failed to fetch applicant review: " + res.statusText);
                  throw new Error("Failed to fetch applicant review");
                }
                return res.json();
              })
              .then(data => {
                document.getElementById("hr-review").innerText = data.review;
              })
              .catch(err => {
                console.error("Applicant review error:", err);
              });
          };
          ul.appendChild(li);
        });
      })
      .catch(err => {
        console.error("Applicants fetch error:", err);
      });
  }
}
