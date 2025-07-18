const BASE_URL = "https://irsas.onrender.com"; 

// Login logic for both existing and new users
function login() {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const role = document.getElementById("role").value;

  if (!email || !password) {
    alert("Please enter email and password");
    return;
  }

  const form = new FormData();
  form.append("email", email);
  form.append("password", password);
  if (role) form.append("role", role);

  fetch(BASE_URL + "/login", { method: "POST", body: form })
    .then(res => {
      if (!res.ok) return res.text().then(text => { throw new Error(text); });
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
      alert("Login failed: " + err.message);
    });
}

// Upload CV, which is parsed and passed into the ML model for review
function uploadCV() {
  const file = document.getElementById("cv").files[0];
  if (!file) {
    alert("Please select a file to upload.");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("token", localStorage.getItem("token"));

  // File gets parsed (textract), parsed text goes to ML model, review is stored
  fetch(BASE_URL + "/upload", { method: "POST", body: form })
    .then(res => {
      if (!res.ok) {
        return res.text().then(msg => { throw new Error(msg); });
      }
      return res.json();
    })
    .then(data => {
      // Shows the final ML-based review (not the raw parsed text)
      document.getElementById("result").innerText = data.review;
    })
    .catch(err => {
      alert("Upload failed: " + err.message);
      console.error("Upload CV error:", err);
    });
}

// Retrieves latest AI review for applicant
function getReview() {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("You need to login first.");
    return;
  }

  fetch(`${BASE_URL}/review?token=${token}`)
    .then(res => {
      if (!res.ok) {
        return res.text().then(msg => { throw new Error(msg); });
      }
      return res.json();
    })
    .then(data => {
      document.getElementById("result").innerText = data.review;
    })
    .catch(err => {
      alert("Review fetch failed: " + err.message);
      console.error("Get review error:", err);
    });
}

// HR dashboard: show list of applicants, click to view their reviews
if (window.location.pathname.includes("hr.html")) {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("You need to login first.");
    location.href = "index.html";
  } else {
    fetch(`${BASE_URL}/applicants?token=${token}`)
      .then(res => {
        if (!res.ok) return res.text().then(text => { throw new Error(text); });
        return res.json();
      })
      .then(applicants => {
        const ul = document.getElementById("applicants");
        ul.innerHTML = "";
        if (applicants.length === 0) {
          ul.innerHTML = "<li>No applicants found.</li>";
          return;
        }
        applicants.forEach(email => {
          const li = document.createElement("li");
          li.textContent = email;
          li.onclick = () => {
            fetch(`${BASE_URL}/applicant_review?email=${encodeURIComponent(email)}&token=${token}`)
              .then(res => {
                if (!res.ok) return res.text().then(text => { throw new Error(text); });
                return res.json();
              })
              .then(data => {
                document.getElementById("hr-review").innerText = data.review;
              })
              .catch(err => {
                console.error("Applicant review error:", err);
                alert("Failed to load applicant review: " + err.message);
              });
          };
          ul.appendChild(li);
        });
      })
      .catch(err => {
        console.error("Applicants fetch error:", err);
        alert("Failed to load applicant list: " + err.message);
      });
  }
}
