document.getElementById('predictButton').addEventListener('click', function(e) {
    e.preventDefault();
    var team1 = document.getElementById('team1').value;
    var team2 = document.getElementById('team2').value;
    // Make an AJAX request to the Python server
    fetch(`http://localhost:5501/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ team1: team1, team2: team2 })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Server returned an error response');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('outcome').innerHTML = 'Prediction: ' + data.outcome;
        document.getElementById('probability').innerHTML = 'Prediction: ' + data.probability;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

var teamLogos = {
    "AFCBournemouth": "/Premier_pic/AFC_Bournemouth.png",
    "Arsenal": "/Premier_pic/Arsenal_FC.png",
    "AstonVilla": "/Premier_pic/Aston_Villa_FC_logo.png",
    "BirminghamCity": "/Premier_pic/Birmingham_City_FC_logo.png",
    "BlackburnRovers": "/Premier_pic/Blackburn_Rovers.png",
    "Blackpool": "/Premier_pic/Blackpool_FC_logo.png",
    "BoltonWanderers": "/Premier_pic/Bolton_Wanderers_FC_logo.png",
    "BrightonAndHoveAlbion": "/Premier_pic/Brighton_&_Hove_Albion_logo.png",
    "Burnley": "/Premier_pic/Burnley_FC_Logo.png",
    "CardiffCity": "/Premier_pic/Cardiff_City_crest.png",
    "CharltonAthletic": "/Premier_pic/CharltonBadge_30Jan2020.png",
    "Chelsea": "/Premier_pic/Chelsea_FC.png",
    "CrystalPalace": "/Premier_pic/Crystal_Palace_FC_logo_(2022).png",
    "DerbyCounty": "/Premier_pic/Derby_County_crest.png",
    "Everton": "/Premier_pic/Everton_FC_logo.png",
    "Fulham": "/Premier_pic/Fulham_FC_(shield).png",
    "HuddersfieldTown": "/Premier_pic/Huddersfield_Town_A.F.C._logo.png",
    "HullCity": "/Premier_pic/Hull_City_A.F.C._logo.png",
    "LeicesterCity": "/Premier_pic/Leicester_City_crest.png",
    "Liverpool": "/Premier_pic/Liverpool_FC.png",
    "ManchesterCity": "/Premier_pic/Manchester_City_FC_badge.png",
    "ManchesterUnited": "/Premier_pic/Manchester_United_FC_crest.png",
    "Middlesbrough": "/Premier_pic/Middlesbrough_FC_crest.png",
    "NewcastleUnited": "/Premier_pic/Newcastle_United_Logo.png",
    "NorwichCity": "/Premier_pic/Norwich_City_FC_logo.png",
    "Portsmouth": "/Premier_pic/Portsmouth_FC_logo.png",
    "QueensParkRangers": "/Premier_pic/Queens_Park_Rangers_crest.png",
    "Reading": "/Premier_pic/Reading_FC.png",
    "SheffieldUnited": "/Premier_pic/Sheffield_United_FC_logo.png",
    "Southampton": "/Premier_pic/FC_Southampton.png",
    "StokeCity": "/Premier_pic/Stoke_City_FC.png",
    "Sunderland": "/Premier_pic/Logo_Sunderland.png",
    "SwanseaCity": "/Premier_pic/Swansea_City_AFC_logo.png",
    "TottenhamHotspur": "/Premier_pic/Tottenham_Hotspur.png",
    "Watford": "/Premier_pic/Watford.png",
    "WestBromwichAlbion": "/Premier_pic/West_Bromwich_Albion.png",
    "WestHamUnited": "/Premier_pic/West_Ham_United_FC_logo.png",
    "WiganAthletic": "/Premier_pic/Wigan_Athletic.png",
    "WolverhamptonWanderers": "/Premier_pic/Wolverhampton_Wanderers.png"
    // Add other teams and their logo paths
};

// Function to update team logo
function updateTeamLogo(teamSelectId, logoImgId) {
    var teamName = document.getElementById(teamSelectId).value;
    var logoUrl = teamLogos[teamName];
    var imgElement = document.getElementById(logoImgId);
    
    if (logoUrl) {
        imgElement.src = logoUrl;
        imgElement.style.display = 'block';
    } else {
        imgElement.style.display = 'none';
    }
}

// Event listeners for team selection changes
document.getElementById('team1').addEventListener('change', function() {
    updateTeamLogo('team1', 'team1Logo');
});

document.getElementById('team2').addEventListener('change', function() {
    updateTeamLogo('team2', 'team2Logo');
});

