window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "static/interpolation_imgs/zebra_surfboard_108";
// var NUM_INTERP_FRAMES = 50;

var frame_ls = [50, 50, 8, 8, 4, 4]

var inter_img_dir_ls = [
  // timesteps imgs
  'static/interpolation_imgs/zebra_surfboard_108',
  'static/interpolation_imgs/donut_couch_119',
  // head imgs
  'static/interpolation_imgs/head_imgs/person_dining_table_4449375153031870133',
  'static/interpolation_imgs/head_imgs/pizza_umbrella_4144782314998949497',
  // resolution imgs
  'static/interpolation_imgs/res_imgs/apple_bowl_8992446158803658395',
  'static/interpolation_imgs/res_imgs/bench_giraffe_4499098420274898487',
]

var interpolate_name_ls = [
  ['#interpolation-slider', '#interpolation-image-wrapper', "#interpolation-slider-text"],
  ['#interpolation-slider-1', '#interpolation-image-wrapper-1', "#interpolation-slider-text-1"],
  ['#interpolation-slider-2', '#interpolation-image-wrapper-2', "#interpolation-slider-text-2"],
  ['#interpolation-slider-3', '#interpolation-image-wrapper-3', "#interpolation-slider-text-3"],
  ['#interpolation-slider-4', '#interpolation-image-wrapper-4', "#interpolation-slider-text-4"],
  ['#interpolation-slider-5', '#interpolation-image-wrapper-5', "#interpolation-slider-text-5"],
]

var inter_img_ls = [];

var interp_images = [];
function preloadInterpolationImages() {

  for (var k = 0; k < inter_img_dir_ls.length; k++) {
    var inter_img_dir = inter_img_dir_ls[k];
    var inter_img_ls_k = [];
    var frame_ls_k = frame_ls[k]; // 50 or 8 
    for (var i = 0; i < frame_ls_k; i++) {
      // deal with different img name format
      var img_idx = i;
      if (frame_ls_k == 50) {
        img_idx = i + 1;
      }
      var path = inter_img_dir + '/img' + String(img_idx).padStart(3, '0') + '.png';
      var image = new Image();
      image.src = path;
      inter_img_ls_k[i] = image;
    }
    inter_img_ls[k] = inter_img_ls_k;
  }
}

function setInterpolationImage(img_ls_idx, i) {
  var image = inter_img_ls[img_ls_idx][i];

  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };

  var inter_warper_name = interpolate_name_ls[img_ls_idx][1];

  // change image here
  $(inter_warper_name).empty().append(image);

  var inter_slider_text_name = interpolate_name_ls[img_ls_idx][2];

  

  var display_text;

  if (frame_ls[img_ls_idx] == 50) {
    // timestep interpolation
    display_text = "DDIM Step: " + String(Number(i) + Number(1)) + " / 50";
  }else if (frame_ls[img_ls_idx] == 8) {
    // head interpolation
    display_text = "Head: " + String(Number(i) + Number(1));
  }else if (frame_ls[img_ls_idx] == 4) {
    // res interpolation
    if (i == 0) {
      display_text = "Middle Block 8x8";
    }else if (i == 1) {
      display_text = "Decoder 16x16";
    }else if (i == 2) {
      display_text = "Decoder 32x32";
    }else if (i == 3) {
      display_text = "Decoder 64x64";
    }else{
      display_text = "Unknown Res"
    }
  }else{
    display_text = "Unknown"
  }
  
  $(inter_slider_text_name).text(display_text)
}

$(document).ready(function() {

    // collapsible
    var coll = document.getElementsByClassName("collapsible");

    for (var i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var collapse_content = this.nextElementSibling;

        collapse_content.classList.toggle("collapse_hide")
      });
    }

    // carousel
    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

  
    for (var i = 0; i < interpolate_name_ls.length; i++) {
      
      // var img_name_idx = i;
      // console.log(img_name_idx)

      (function(i) {
        // console.log(i)
        var inter_slider_name = interpolate_name_ls[i][0];
        // console.log(inter_slider_name)
        $(inter_slider_name).on('input', function(event) {
          // console.log(this.value)
          setInterpolationImage(i, this.value);
        })

        setInterpolationImage(i, 0);

        var frame_ls_i = frame_ls[i];

        $(inter_slider_name).prop('max', frame_ls_i - 1);
      }(i));


    }

    bulmaSlider.attach();
    
})
function changeSize(){
  const height= $('#resize_image1').height()
  const width= $('#resize_image1').width()
  $('#row_1_cake').height(height)
  $('#row_1_cake').width(width)
  $('#row_1_keyboard').height(height)
  $('#row_1_keyboard').width(width)
}
changeSize()
 window.addEventListener('resize', function(event) {
      changeSize()
      if(height/width>1.5){
        window.location.reload()
      }
}, false);

// $(window).resize(function() {
//   //resize just happened, pixels changed
//   console.log("window resize")
//   location.reload();
// });


