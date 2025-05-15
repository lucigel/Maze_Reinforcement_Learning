import React from 'react';
import { NavLink } from 'react-router-dom';
import '../../styles/Navigation.css';

const Navigation = () => {
  return (
    <nav className="navigation">
      <ul className="nav-links">
        <li>
          <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>
            Sinh mê cung
          </NavLink>
        </li>
        <li>
          <NavLink to="/solver" className={({ isActive }) => isActive ? 'active' : ''}>
            Giải mê cung
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Navigation;