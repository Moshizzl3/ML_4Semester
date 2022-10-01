async function getFromUrl(url) {
  const response = await fetch(url);
  const data = await response.json();
 

  return data;
}

async function buildList() {
  const apiKey = 'RGAPI-8b194425-8073-4af5-aa5e-180233fa5c2e'
  const data = await getFromUrl(
    "https://europe.api.riotgames.com/lol/match/v5/matches/EUW1_6005832559?api_key="+apiKey
  );
  const gameInfo = data.info;
  //Teamid 100 = blue
  const result = {
    blueGold: gameInfo.participants
      .filter((x) => x.teamId == 100)
      .reduce((total, thing) => total + thing.goldEarned, 0),
    blueMinionsKilled: gameInfo.participants
      .filter((x) => x.teamId == 100)
      .reduce((total, thing) => total + thing.totalMinionsKilled, 0),
    blueJungleMinionsKilled: gameInfo.participants
      .filter((x) => x.teamId == 100)
      .reduce(
        (total, thing) => total + thing.challenges.alliedJungleMonsterKills,
        0
      ),
    blueAvgLevel:
      gameInfo.participants
        .filter((x) => x.teamId == 100)
        .reduce((total, thing) => total + thing.champLevel, 0) / 5,

    redGold: gameInfo.participants
      .filter((x) => x.teamId == 200)
      .reduce((total, thing) => total + thing.goldEarned, 0),
    redMinionsKilled: gameInfo.participants
      .filter((x) => x.teamId == 200)
      .reduce((total, thing) => total + thing.totalMinionsKilled, 0),
    redJungleMinionsKilled: gameInfo.participants
      .filter((x) => x.teamId == 200)
      .reduce(
        (total, thing) => total + thing.challenges.alliedJungleMonsterKills,
        0
      ),
    redAvgLevel:
      gameInfo.participants
        .filter((x) => x.teamId == 200)
        .reduce((total, thing) => total + thing.champLevel, 0) / 5,

    blueChampKills: gameInfo.participants
      .filter((x) => x.teamId == 100)
      .reduce((total, thing) => total + thing.kills, 0),
    blueHeraldKills: gameInfo.teams[0].objectives.riftHerald.kills,
    blueDragonKills: gameInfo.teams[0].objectives.dragon.kills,
    blueTowersDestroyed: gameInfo.teams[0].objectives.tower.kills,

    redChampKills: gameInfo.participants
      .filter((x) => x.teamId == 200)
      .reduce((total, thing) => total + thing.kills, 0),
    redHeraldKills: gameInfo.teams[1].objectives.riftHerald.kills,
    redDragonKills: gameInfo.teams[1].objectives.dragon.kills,
    redTowersDestroyed: gameInfo.teams[1].objectives.tower.kills,
  };

  for(let i in result){
    console.log(i)
  }

  console.log(gameInfo.teams[0].teamId, "Teamid:")
  console.log(gameInfo.teams[0].win, "teamwin")

console.log(Object.values(result))

}

buildList();


async function test(){
  const apiKey = 'RGAPI-8b194425-8073-4af5-aa5e-180233fa5c2e'
  const data = await getFromUrl(
    "https://europe.api.riotgames.com/lol/match/v5/matches/EUW1_6005832559/timeline?api_key="+apiKey
  );
 
  console.log(data)

}

test()