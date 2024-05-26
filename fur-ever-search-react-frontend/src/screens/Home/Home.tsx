import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

import './Home.css';
import { MainNavigation } from '../../components/MainNavigation';
import { sendUserInteraction } from '../../helpers/userInteraction';
import { formattedBreedsMock } from '../../data/breeds';

export interface ILandings {
  landings: any
}

const Home: React.FC<any> = (props) => {

  const [breeds, setBreeds] = useState<any[]>([]);
  const [filteredBreeds, setFilteredBreeds] = useState<any[]>([]);
  const [keyword, setKeyword] = useState('');
  const [showUserSpecificBreeds, setShowUserSpecificBreeds] = useState(true);

  const storedData = sessionStorage.getItem('mySessionData') || '';
  const retrievedSessionData = storedData && JSON.parse(storedData);

  useEffect(() => {
    const normalizedKeyword = keyword.toLowerCase();
    const filteredBreeds = breeds.filter((breed: any) => {
      const normalizedBreedName = breed.name.toLowerCase();
      const breedNameWords = normalizedBreedName.split(' ');

      // Check if any word in breed name starts with the keyword
      const matchesKeyword = breedNameWords.some((word: any) => word.startsWith(normalizedKeyword));

      return keyword ? matchesKeyword : true;
    });

    setFilteredBreeds(filteredBreeds);
  }, [keyword, breeds]);

  useEffect(() => {
    console.log('retrievedSessionData - jashp', retrievedSessionData);
    if (showUserSpecificBreeds && retrievedSessionData?.userInfo && retrievedSessionData?.recommendedBreedsData) {
      const formattedBreeds = retrievedSessionData?.recommendedBreedsData;
      console.log('formattedBreeds - jashp', formattedBreeds);
      setBreeds(formattedBreeds);
      setFilteredBreeds(formattedBreeds);
    } else {
      // Fetch data from the API endpoint
      axios.get('http://127.0.0.1:5000/breeds')
        .then(response => {
          // console.log('response', response.data)
          // Extract breed name, description, and breed image URL
          // Parse the JSON string into a JavaScript object
          const responseData = response?.data
          console.log('responseData - jashp', responseData.breeds)
          const formattedBreeds = responseData.breeds.map((breed: any) => ({
            name: breed['\u00ef\u00bb\u00bfbreed'],
            description: breed.description,
            imageUrls: JSON.parse(breed.images.replace(/'/g, "\""))
          }));
          console.log('formattedBreeds - jashp', formattedBreeds)
          // Set the formatted breed data to the state
          setBreeds(formattedBreeds);
          setFilteredBreeds(formattedBreeds);
        })
        .catch(error => {
          console.error('Error fetching data:', error);
          console.log('using local data'); 
          setBreeds(formattedBreedsMock);
          setFilteredBreeds(formattedBreedsMock);
        });
    }
  }, [showUserSpecificBreeds]);

  const switchType = () => {
    setShowUserSpecificBreeds(!showUserSpecificBreeds);
  }

  const breedClick = (breed: any) => {
    const uid = retrievedSessionData?.userInfo?.user_id;
    if (uid) {
      sendUserInteraction(uid, breed.name, 'Clicked');
      if (keyword) {
        sendUserInteraction(uid, breed.name, 'ManualSearch');
      }
    }
  }

  console.log('breeds - jashp', breeds)

  return (
    <div className='App'>
      <MainNavigation cartItemNumber={0} />
      <h1>{retrievedSessionData && showUserSpecificBreeds ? `Recommended Dogs for ${retrievedSessionData?.userInfo?.firstname}` : 'All Available Dogs'}</h1>
      <div className="search-container">
        <input className="search-input" type="text" placeholder="Start typing for breed..." value={keyword} onChange={(e) => setKeyword(e.target.value)} />
        {retrievedSessionData && <button className="search-button" style={{ marginLeft: '10px' }} onClick={switchType}>{!showUserSpecificBreeds ? 'My Recommendations' : 'All Available Dogs'}</button>}
      </div>
      <br />
      <br />
      <div className="dog-container">
        {filteredBreeds.map((breed: any, index: number) => (
          breed.imageUrls[0] &&
          <Link to={`/fur-ever-search-app/breed-details/${breed.name}`} className="dog-card" key={index} onClick={() => breedClick(breed)}>
            <img className="dog-image" src={breed.imageUrls[0]} alt={`Dog ${index + 1}`} />
            <p>{breed.name}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default Home;
