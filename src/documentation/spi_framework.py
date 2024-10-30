"""SPI Framework Documentation for CIRCMAN5.0"""

class SPIFramework:
    def __init__(self):
        self.hierarchical_levels = {
            'factory': ['energy efficiency', 'resource utilization'],
            'manufacturing': ['process optimization', 'waste reduction'],
            'supply_chain': ['material flow', 'logistics'],
            'product': ['design for circularity', 'end-of-life']
        }
        
        self.evaluation_criteria = {
            'environmental_impact': ['emissions', 'resource consumption'],
            'circularity': ['recycled content', 'recyclability'],
            'efficiency': ['yield rate', 'energy usage']
        }

    def document_methodology(self):
        return {
            'evaluation_approach': self._define_evaluation_approach(),
            'assessment_criteria': self._define_assessment_criteria()
        }