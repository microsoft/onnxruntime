$(window).scroll(function () {
  $("header").toggleClass("scrolled", $(this).scrollTop() > 50);
  backTop();
});
$(document).ready(function () {
  equalHeight();
  backTop();

  $("button.navbar-toggler").click(function (event) {
    $("header").toggleClass("header-collapse");
  });
  $("header").toggleClass("scrolled", $(this).scrollTop() > 50);

  // scroll body to 0px on click
  $("#back-to-top").click(function () {
    $("body,html").animate(
      {
        scrollTop: 0,
      },
      400
    );
    $("#ONNXLogo").focus();
    return false;
  });

  $(document).click(function (event) {
    var clickover = $(event.target);
    var _opened = $(".navbar-collapse").hasClass("show");
    if (_opened === true && !clickover.hasClass("navbar-toggler")) {
      $(".navbar-toggler").click();
    }
  });

  $("#listbox-5").focus(function () {
    var top = $(".get-started-section").offset().top;
    $(window).scrollTop(top);
  });

  $(document).keyup(function (e) {
    if ($("#navbarNav").hasClass("show")) {
      if (e.keyCode === 27) $("button.navbar-toggler").click(); // esc
    }
  });

  $(".btn-getStarted").click(function () {
    var tableTop = $("#getStartedTable").offset().top;
    $("body,html").animate(
      {
        scrollTop: tableTop - 100,
      },
      600
    );
   
  });
  $(document).on("click", ".resources-img", function(e) {
    var data = $(this).attr('data-src');
    var iframe = '<iframe width="100%" height="400" title="Resources" src="https://www.youtube.com/embed/'+data+'?autoplay=1" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
    $('#myModal .modal-body').html(iframe);
    $(".btn-modal").click();
  });
  $("#myModal").on('hide.bs.modal', function () {
    $('#myModal .modal-body').children('iframe').attr('src', '');
  });
  $(document).on("focus", function (e) {
    var docTop = $(this).offset().top;
    $(window).scrollTop(docTop);
  });

  getStartedScroll();
  carouselNormalization();

  $('.nav-tabs').responsiveTabs();
});

function getStartedScroll() {
  var windowsHash = location.hash.split("#");
  var tabelId = $("#getStartedTable");
  if (tabelId.length) {
    var tableTop = tabelId.offset().top;
  }
  if (windowsHash[1] === "getStartedTable") {
    $("body,html").animate(
      {
        scrollTop: tableTop - 100,
      },
      600
    );
  }
}

$(window).resize(function () {
  equalHeight();
});

function backTop() {
  if ($(this).scrollTop() > 50) {
    $("#back-to-top").fadeIn();
  } else {
    $("#back-to-top").fadeOut();
  }
}

function equalHeight() {
  if (window.innerWidth > 767) {
    var maxHeight = 0;
    $(".equalHeight h2").height("auto");
    $(".equalHeight h2").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight h2").height(maxHeight);

    var maxHeight = 0;
    $(".equalHeight p").height("auto");
    $(".equalHeight p").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight p").height(maxHeight);

    var maxHeight = 0;
    $(".equalHeight-1 h3").height("auto");
    $(".equalHeight-1 h3").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight-1 h3").height(maxHeight);

    var maxHeight = 0;
    $(".equalHeight-1 .onnx-model-content").height("auto");
    $(".equalHeight-1 .onnx-model-content").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight-1 .onnx-model-content").height(maxHeight);

    var maxHeight = 0;
    $(".equalHeight-2 h3").height("auto");
    $(".equalHeight-2 h3").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight-2 h3").height(maxHeight);

    var maxHeight = 0;
    $(".equalHeight-2 p.first-child").height("auto");
    $(".equalHeight-2 p.first-child").each(function () {
      if ($(this).height() > maxHeight) {
        maxHeight = $(this).height();
      }
    });
    $(".equalHeight-2 p.first-child").height(maxHeight);
  } else {
    $(".equalHeight h2").height("auto");
    $(".equalHeight p").height("auto");
    $(".equalHeight-1 h3").height("auto");
    $(".equalHeight-1 .onnx-model-content").height("auto");
    $(".equalHeight-2 h3").height("auto");
    $(".equalHeight-2 p.first-child").height("auto");
  }
}

function carouselNormalization() {
  var items = $("#ONNXCarousel .item"),
    heights = [],
    tallest;
  if (items.length) {
    function normalizeHeights() {
      items.each(function () {
        heights.push($(this).height());
      });
      tallest = Math.max.apply(null, heights);
      items.each(function () {
        $(this).css("min-height", tallest + "px");
      });
    }
    normalizeHeights();
    $(window).on("resize orientationchange", function () {
      (tallest = 0), (heights.length = 0);
      items.each(function () {
        $(this).css("min-height", "0");
      });
      normalizeHeights();
    });
  }
}

(function ($){
	$.fn.responsiveTabs = function() {
	this.addClass('ddTabs'),
	this.append($('<span class="dropdown-arrow"></span>')),

	this.on("click", "li > a.active, span.dropdown-arrow", function (){
			this.toggleClass('open');
		}.bind(this)), this.on("click", "li > a:not(.active)", function() {
	        this.removeClass("open")
	    }.bind(this)); 
	}
})(jQuery);