"use strict";

// get all of the releases from versions.json, and use these to populate the
// dropdown menu of different releases
$(document).ready(function () {
    // Define versions_json_url in versions.html through a template variable
    $.getJSON(versions_json_url)
        .done(function (data) {
            $.each(data.sort(function (a, b) {
                return a.version > b.version
            }), function (i, item) {
                $('<li class="list-inline-item version">').append(
                    $("<a>").text(item.display).attr('href', item.url)
                ).appendTo("#other-versions");
            });
        })
        .fail(function (d, textStatus, error) {
            console.error("getJSON failed, status: " + textStatus + ", error: " + error);
        });
});

console.log("Loading versions from " + versions_json_url);
