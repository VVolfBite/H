package net.lab1024.sa.admin.module.system.earthquake.dao;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.earthquake.domain.entity.EarthquakeEntity;
import net.lab1024.sa.admin.module.system.earthquake.domain.form.EarthquakeQueryForm;
import net.lab1024.sa.admin.module.system.earthquake.domain.vo.EarthquakeVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;

@Mapper
@Component
public interface EarthquakeDao extends BaseMapper<EarthquakeEntity> {
    /**
     * 查询震情列表
     *
     * @param page
     * @param queryForm
     * @return
     */
    List<EarthquakeVO> queryEarthquake(Page page, @Param("queryForm") EarthquakeQueryForm queryForm);

    /**
     * 查询震情
     */
    List<EarthquakeVO> selectEarthquakeByDeleted( @Param("deletedFlag") Boolean deletedFlag);

    /**
     * 通过地址码查询
     *
     * @param geoCode
     * @return
     */
    String getByGeoCode(@Param("geoCode") String geoCode);

    /**
     * 通过震情码查询
     *
     * @param code
     * @return
     */
    EarthquakeEntity getByCode(@Param("code") String code);


    /**
     * 获取所有震情信息
     *
     * @return
     */
    List<EarthquakeVO> listAll();

    /**
     * 获取一批震情
     *
     * @param codes
     * @return
     */
    List<EarthquakeVO> getEarthquakeByCodes(@Param("codes") Collection<String> codes);

    /**
     * 查询单个震情信息
     *
     * @param code
     * @return
     */
    EarthquakeVO getEarthquakeByCode(@Param("code") String code);

    /**
     * 获取所有
     *
     * @param leaveFlag
     * @return
     */
    List<String> getEarthquakeCode(@Param("leaveFlag") Boolean leaveFlag);

}
