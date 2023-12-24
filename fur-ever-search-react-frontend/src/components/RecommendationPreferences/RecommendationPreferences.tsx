import React, { useState } from "react";
import Select from 'react-select';
import "./RecommendationPreferences.css"; // Import the CSS file for styling
import axios from "axios";
import { recommendationOptions, recommendationSubGroups, breed_features } from './helper'

export const RecommendationPreferences = ({ breedName, setRecommendationPreferences }: any) => {
  const onSelectChange = (selections: any) => {
    let selectedPreferences = selections.map((selection: any) => {
      return selection.label
    })
    let allSelectedPreferences: any = []
    selectedPreferences.forEach((preference: any) => {
      if (preference == 'Coat') {
        allSelectedPreferences = [...allSelectedPreferences, 'Coat Type', 'Coat Length']
      } else if(preference == 'Physical') {
        allSelectedPreferences = [...allSelectedPreferences, 'min_height', 'max_height', 'min_weight', 'max_weight']
      } else if(preference == 'Expectancy') {
        allSelectedPreferences = [...allSelectedPreferences, 'min_expectancy', 'max_expectancy']
      } else {
        allSelectedPreferences = [...allSelectedPreferences, preference]
      }
    });
    setRecommendationPreferences(allSelectedPreferences)
  }

  const arrayToObjectList = breed_features.map(feature => ({
    value: feature,
    label: feature,
  }));

  // const convertedList = Object.entries(recommendationSubGroups).map(([heading, options]: any) => ({
  //   label: heading,
  //   options: options.map((option: any) => ({
  //     label: option,
  //     value: option,
  //   }))
  // }));

  // console.log('convertedList - jashp', convertedList)

  return (
    <div className="sticky-header-recommendations">
      <h2 className="section-name">Similar to {breedName}</h2>
      <div className="preferences-select">
        <Select
          isMulti
          options={arrayToObjectList}
          className="basic-multi-select"
          classNamePrefix="select"
          placeholder="Select Preferences..."
          onChange={onSelectChange}
        />
      </div>
      {/* <br /> */}
      {/* <button className="find-button" onClick={() => { }}>
        Get Recommendations by Preferences
      </button> */}
    </div>
  );
};