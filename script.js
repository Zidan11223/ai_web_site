'use strict';


/*preload*/
const preloader = document.querySelector("[data-preload]");
window.addEventListener("load", function(){
    preloader.classList.add("loaded");
    document.body.classList.add("loaded");
});


/*add event listener on multiple elements*/
const addEventOneElements = function(elements, evenType, callback){
    for(let i = 0, len = elements.length; i< len; i++){
        elements[i].addEventListener(evenType, callback)
    }
}


/*navbar*/
const navbar = document.querySelector("[data-navbar]");
const navTogglers = document.querySelectorAll("[data-nav-toggler]");
const overlay = document.querySelector("[data-overlay]");
const toggleNavbar = function(){
    navbar.classList.toggle("active");
    overlay.classList.toggle("active");
    document.body.classList.toggle("nav-active");
}
addEventOneElements(navTogglers, "click", toggleNavbar);
const navbarLinks = document.querySelectorAll("[data-nav-link]");
const closeNavbar = function () {
  navbar.classList.remove("active");
  overlay.classList.remove("active");
  document.body.classList.remove("nav-active");
}
addEventOneElements(navbarLinks, "click", closeNavbar);


/*header and back top btn*/
const header = document.querySelector("[data-header]");
const backTopBtn = document.querySelector("[data-back-top-btn]");
let lastScrollPos = 0;
const hideHeader = function () {
    const isScrollBottom = lastScrollPos < window.scrollY;
    if(isScrollBottom){
        header.classList.add("hide");
    }
    else{
        header.classList.remove("hide");
    }
    lastScrollPos = window.scrollY;
}
window.addEventListener("scroll", function(){
    if(window.scrollY >= 50){
        header.classList.add("active");
        backTopBtn.classList.add("active");
        hideHeader();
    }
    else{
        header.classList.remove("active");
        backTopBtn.classList.remove("active");
    }
});


/*hero slider*/
const heroSlider = document.querySelector("[data-hero-slider]");
const heroSliderItems = document.querySelectorAll("[data-hero-slider-item]");
const heroSliderPrevBtn = document.querySelector("[data-prev-btn]");
const heroSliderNextBtn = document.querySelector("[data-next-btn]");
let currentSliderPos = 0;
let lastActiveSliderItem = heroSliderItems[0];
const updateSliderPos = function(){
    lastActiveSliderItem.classList.remove("active");
    heroSliderItems[currentSliderPos].classList.add("active");
    lastActiveSliderItem = heroSliderItems[currentSliderPos];
}
const slideNext = function(){
    if(currentSliderPos >= heroSliderItems.length - 1){
    currentSliderPos = 0;
    }
    else{
        currentSliderPos++;
    }
    updateSliderPos();
}
heroSliderNextBtn.addEventListener("click", slideNext);
const slidePrev = function(){
    if(currentSliderPos <= 0){
        currentSliderPos = heroSliderItems.length - 1;
    }
    else{
        currentSliderPos--;
    }
    updateSliderPos();   
}
heroSliderPrevBtn.addEventListener("click", slidePrev);


/*auto slide*/
let autoSliderInterval;

const autoSlide = function(){
    autoSliderInterval = setInterval(function(){
        slideNext(); 
    }, 7000);
}
addEventOneElements([heroSliderNextBtn, heroSliderPrevBtn], "mouseover", function(){
    clearInterval(autoSliderInterval);
});
addEventOneElements([heroSliderNextBtn, heroSliderPrevBtn], "mouseout", autoSlide);
window.addEventListener("load", autoSlide);


/*parallax efect*/
const parallaxItems = document.querySelectorAll("[data-parallax-item]");
let x, y;
window.addEventListener("mousemove", function (event) {
    x = (event.clientX / window.innerWidth * 10) - 5;
    y = (event.clientY / window.innerHeight * 10) - 5;
    x = x - (x * 2);
    y = y - (y * 2);
    for(let i = 0, len = parallaxItems.length; i < len; i++){
        x = x * Number(parallaxItems[i].dataset.parallaxSpeed);
        y = y * Number(parallaxItems[i].dataset.parallaxSpeed);
        parallaxItems[i].style.transform = `translate3d(${x}px, ${y}px, 0px)`; 
    }
});


/*reservation*/
var form = document.getElementById("sheetdb-form");
form.addEventListener("submit", function(e) {
e.preventDefault();
fetch(form.action, {
method: "POST",
body: new FormData(form),
})
.then(response => response.json())
.then((data) => {
alert("Reservation Submitted! Thank you " + document.getElementById("uname").value);
form.reset();
});
});